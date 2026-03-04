"""
scripts/train_flowmatching.py
==============================
Training TCFlowMatching v4 với OT-CFM + PINN Vorticity.

Metrics:
  - ADE/FDE/24h/48h/72h (km)
  - Train time per epoch (s)
  - Sample time per batch (ms)
  - Loss breakdown: fm, dir, smooth, disp, curv, pinn(NS)

Data split:
  - train/  : training data
  - val/    : validation (early stopping, best model selection)
  - test/   : held-out test set (chỉ đánh giá cuối cùng, KHÔNG dùng để tune)

Chạy:
  python scripts/train_flowmatching.py \
      --dataset_root TCND_vn \
      --output_dir   model_save/flowmatching_v4 \
      --ode_steps 10 --sigma_min 0.001 \
      --num_epochs 200 --batch_size 32
"""

import argparse, os, sys, time, math
import torch, torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TCNM.data.loader import data_loader
from TCNM.flow_matching_model import TCFlowMatching
from TCNM.utils import get_cosine_schedule_with_warmup


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root',        default='TCND_vn',                  type=str)
    p.add_argument('--obs_len',             default=8,                           type=int)
    p.add_argument('--pred_len',            default=12,                          type=int)
    p.add_argument('--batch_size',          default=32,                          type=int)
    p.add_argument('--num_epochs',          default=200,                         type=int)
    p.add_argument('--g_learning_rate',     default=2e-4,                        type=float)
    p.add_argument('--weight_decay',        default=1e-4,                        type=float)
    p.add_argument('--warmup_epochs',       default=5,                           type=int)
    p.add_argument('--grad_clip',           default=1.0,                         type=float)
    p.add_argument('--patience',            default=40,                          type=int)
    p.add_argument('--gpu_num',             default='0',                         type=str)
    p.add_argument('--output_dir',          default='model_save/flowmatching_v4',type=str)
    p.add_argument('--save_interval',       default=10,                          type=int)
    p.add_argument('--test_year',           default=2019,                        type=int,
                   help='Năm để filter test set (held-out, chỉ eval cuối)')
    p.add_argument('--val_ensemble',        default=5,                           type=int)
    p.add_argument('--ode_steps',           default=10,                          type=int,
                   help='ODE integration steps cho Flow Matching')
    p.add_argument('--val_freq',            default=5,                           type=int)
    p.add_argument('--sigma_min',           default=0.001,                       type=float)
    # compat
    p.add_argument('--d_model',    default=128,   type=int)
    p.add_argument('--delim',      default=' ')
    p.add_argument('--skip',       default=1,     type=int)
    p.add_argument('--min_ped',    default=1,     type=int)
    p.add_argument('--threshold',  default=0.002, type=float)
    p.add_argument('--other_modal',default='gph')
    return p.parse_args()


def resolve_data_path(root):
    """
    Trả về (train_dir, val_dir, test_dir).

    Ưu tiên dùng val/ folder riêng biệt.
    Nếu không có val/, fallback về test/ (với cảnh báo).
    """
    root = root.rstrip('/\\')

    # Nếu truyền vào thẳng Data1d/train hoặc tương tự
    if root.endswith(('Data1d/train', 'Data1d\\train')):
        base = root[:-len('train')]
    elif root.endswith(('Data1d/test', 'Data1d\\test')):
        base = root[:-len('test')]
    elif root.endswith('Data1d'):
        base = root + os.sep
    else:
        base = os.path.join(root, 'Data1d') + os.sep

    train_dir = os.path.join(base, 'train')
    val_dir   = os.path.join(base, 'val')
    test_dir  = os.path.join(base, 'test')

    return train_dir, val_dir, test_dir


def move_batch(bl, device):
    for j, x in enumerate(bl):
        if torch.is_tensor(x):
            bl[j] = x.to(device)
        elif isinstance(x, dict):
            bl[j] = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in x.items()}
    return bl


def denorm_traj(n):
    r = n.clone()
    r[..., 0] = n[..., 0] * 50 + 1800
    r[..., 1] = n[..., 1] * 50
    return r


# ── Loss breakdown ────────────────────────────────────────────────────────────
def compute_loss_breakdown(model, batch_list):
    traj_gt = batch_list[1]
    Me_gt   = batch_list[8]
    obs     = batch_list[0]
    obs_Me  = batch_list[7]

    B      = traj_gt.shape[1]
    device = traj_gt.device
    lp, lm = obs[-1], obs_Me[-1]
    sm     = model.sigma_min

    x1    = model.traj_to_rel(traj_gt, Me_gt, lp, lm)
    x0    = torch.randn_like(x1) * sm
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)

    x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
    denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
    target_vel = (x1 - (1 - sm) * x_t) / denom

    pred_vel    = model.net(x_t, t, batch_list)
    fm_loss     = F.mse_loss(pred_vel, target_vel)

    pred_x1     = x_t + denom * pred_vel
    pred_abs, _ = model.rel_to_abs(pred_x1, lp, lm)

    dir_l  = model._dir_loss(pred_abs, traj_gt, lp)
    smt_l  = model._smooth_loss(pred_x1)
    disp_l = model._weighted_disp_loss(pred_abs, traj_gt)
    curv_l = model._curvature_loss(pred_abs, traj_gt)
    pinn_l = model._ns_pinn_loss(pred_abs)

    total = (fm_loss + 2.0*dir_l + 0.5*smt_l
             + 1.0*disp_l + 1.5*curv_l + 0.5*pinn_l)

    return {
        'total':   total,
        'fm':      fm_loss.item(),
        'dir':     dir_l.item(),
        'smooth':  smt_l.item(),
        'disp':    disp_l.item(),
        'curv':    curv_l.item(),
        'ns':      pinn_l.item(),
        'ns_cons': 0.0,
    }


# ── Validation / Test metrics ─────────────────────────────────────────────────
def evaluate_km(model, loader, device, num_ensemble=5,
                ddim_steps=10, pred_len=12):
    model.eval()
    all_step_errors = []
    total_sample_ms = 0.0
    n_batches       = 0

    with torch.no_grad():
        for batch in loader:
            bl = move_batch(list(batch), device)
            gt = bl[1]

            t0 = time.time()
            pred, _ = model.sample(bl, num_ensemble=num_ensemble,
                                   ddim_steps=ddim_steps)
            total_sample_ms += (time.time() - t0) * 1000
            n_batches += 1

            pred_r = denorm_traj(pred)
            gt_r   = denorm_traj(gt)
            dist   = torch.norm(pred_r - gt_r, dim=2) * 11.1
            all_step_errors.append(dist.mean(dim=1).cpu())

    stacked    = torch.stack(all_step_errors, dim=0)
    mean_steps = stacked.mean(dim=0)

    m = {'ADE': mean_steps.mean().item(), 'FDE': mean_steps[-1].item()}
    for h, s in [(12, 1), (24, 3), (48, 7), (72, 11)]:
        if s < pred_len:
            m[f'{h}h'] = mean_steps[s].item()

    m['sample_ms_per_batch'] = total_sample_ms / max(n_batches, 1)
    return m, mean_steps


# ── Best Model Saver ──────────────────────────────────────────────────────────
class BestModelSaver:
    def __init__(self, patience=40, min_delta=2.0, verbose=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter    = 0
        self.best_ade   = float('inf')
        self.early_stop = False

    def __call__(self, ade_km, model, out_dir, epoch, opt, train_loss, val_loss):
        if ade_km < self.best_ade - self.min_delta:
            self.best_ade = ade_km
            self.counter  = 0
            ckpt = os.path.join(out_dir, 'best_model.pth')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state':  opt.state_dict(),
                'train_loss':       train_loss,
                'val_loss':         val_loss,
                'val_ade_km':       ade_km,
                'model_type':       'TCFlowMatching_v4_OT-CFM+PINN',
                'args': {
                    'obs_len':   model.obs_len,
                    'pred_len':  model.pred_len,
                    'sigma_min': model.sigma_min,
                },
            }, ckpt)
            if self.verbose:
                print(f"  ✅ Best val ADE: {ade_km:.1f} km  →  saved {ckpt}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}  "
                      f"(best val ADE={self.best_ade:.1f} km)")
            if self.counter >= self.patience:
                self.early_stop = True


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  TC-FlowMatching v4  |  OT-CFM + PINN Vorticity (NS)")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  ODE steps   : {args.ode_steps}")
    print(f"  sigma_min   : {args.sigma_min}  (OT path, near-deterministic)")
    print(f"  Ensemble    : {args.val_ensemble}")
    print(f"  LR          : {args.g_learning_rate}  WD: {args.weight_decay}")
    print(f"  Epochs      : {args.num_epochs}  Patience: {args.patience}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    train_dir, val_dir, test_dir = resolve_data_path(args.dataset_root)

    print(f"\n  Data paths:")
    print(f"    train : {train_dir}")
    print(f"    val   : {val_dir}")
    print(f"    test  : {test_dir}  (held-out, year={args.test_year})")

    _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)

    # ── Val loader: dùng val/ folder, KHÔNG filter theo year ──────────
    val_loader = None
    if os.path.exists(val_dir):
        _, val_loader = data_loader(args, {'root': val_dir, 'type': 'val'}, test=True)
        print(f"\n  ✅ Val loader: val/ folder  ({len(val_loader.dataset)} seq)")
    else:
        # Fallback: nếu không có val/ thì dùng test/ (cảnh báo)
        print(f"\n  ⚠️  val/ folder không tồn tại! Fallback về test/ (không lý tưởng)")
        if os.path.exists(test_dir):
            _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                        test=True, test_year=args.test_year)
            print(f"  ⚠️  Dùng test/ year={args.test_year} làm val  ({len(val_loader.dataset)} seq)")

    # ── Test loader: held-out, chỉ eval cuối training ─────────────────
    test_loader = None
    if os.path.exists(test_dir):
        _, test_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                     test=True, test_year=args.test_year)
        print(f"  ✅ Test loader: test/ year={args.test_year}  ({len(test_loader.dataset)} seq)")
    else:
        print(f"  ⚠️  test/ folder không tồn tại, bỏ qua final test eval")

    print(f"\n  Train: {len(train_loader.dataset)} seq  "
          f"Val: {len(val_loader.dataset) if val_loader else 0} seq  "
          f"Test: {len(test_loader.dataset) if test_loader else 0} seq\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len  = args.pred_len,
        obs_len   = args.obs_len,
        sigma_min = args.sigma_min,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  : {n_p:,}")
    print(f"  Architecture: VelocityField (OT-CFM + PINN Vorticity)\n")

    assert hasattr(model, "_ns_pinn_loss"), (
        "❌ Model thiếu _ns_pinn_loss! Dùng flow_matching_model_v4.py"
    )
    assert not hasattr(model.net, "ns_physics"), (
        "❌ Model vẫn dùng NavierStokesPhysics MLP (v3)! Dùng flow_matching_model_v4.py"
    )
    print("  ✅ PINN vorticity (NS) confirmed\n")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer    = optim.AdamW(model.parameters(),
                               lr=args.g_learning_rate,
                               weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    saver        = BestModelSaver(patience=args.patience, verbose=True)

    # ── Log files ─────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_ADE_km,val_FDE_km,"
                "val_12h,val_24h,val_48h,val_72h,sigma_min,"
                "epoch_time_s,sample_ms_per_batch,"
                "fm_loss,dir_loss,smooth_loss,disp_loss,curv_loss,pinn_loss,ns_cons\n")

    print("=" * 70 + "\n  TRAINING\n" + "=" * 70)

    epoch_times = []

    for epoch in range(args.num_epochs):

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss  = 0.0
        loss_accum  = {'fm': 0, 'dir': 0, 'smooth': 0,
                       'disp': 0, 'curv': 0, 'ns': 0, 'ns_cons': 0}
        t_epoch_start = time.time()

        for i, batch in enumerate(train_loader):
            bl   = move_batch(list(batch), device)
            bd   = compute_loss_breakdown(model, bl)
            loss = bd['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            for k in loss_accum:
                loss_accum[k] += bd[k]

            if i % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [{epoch:>3}/{args.num_epochs}] [{i:>3}/{len(train_loader)}]"
                      f"  total={loss.item():.4f}"
                      f"  fm={bd['fm']:.3f}"
                      f"  dir={bd['dir']:.3f}"
                      f"  pinn={bd['ns']:.4f}"
                      f"  lr={lr:.2e}")

        epoch_time    = time.time() - t_epoch_start
        epoch_times.append(epoch_time)
        avg_train     = train_loss / len(train_loader)
        n_bat         = len(train_loader)
        avg_breakdown = {k: v / n_bat for k, v in loss_accum.items()}

        # ── Validate trên val/ ─────────────────────────────────────────
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    bl = move_batch(list(batch), device)
                    val_loss += model.get_loss(bl).item()
            avg_val = val_loss / len(val_loader)

            if epoch % args.val_freq == 0 or epoch < 5:
                m, per_step = evaluate_km(
                    model, val_loader, device,
                    num_ensemble=args.val_ensemble,
                    ddim_steps=args.ode_steps,
                    pred_len=args.pred_len,
                )
                ade = m['ADE']
                fde = m['FDE']
                sms = m['sample_ms_per_batch']

                print(f"\n{'─'*70}")
                print(f"  Epoch {epoch:>3}  │  train={avg_train:.4f}  val={avg_val:.4f}"
                      f"  time={epoch_time:.1f}s")
                print(f"  Loss breakdown │"
                      f"  fm={avg_breakdown['fm']:.3f}"
                      f"  dir={avg_breakdown['dir']:.3f}"
                      f"  disp={avg_breakdown['disp']:.3f}"
                      f"  curv={avg_breakdown['curv']:.3f}"
                      f"  pinn(NS)={avg_breakdown['ns']:.4f}")
                print(f"  Val (km)       │"
                      f"  ADE={ade:.1f}  FDE={fde:.1f}"
                      f"  12h={m.get('12h',0):.0f}"
                      f"  24h={m.get('24h',0):.0f}"
                      f"  48h={m.get('48h',0):.0f}"
                      f"  72h={m.get('72h',0):.0f}")
                print(f"  Speed          │"
                      f"  sample={sms:.1f}ms/batch"
                      f"  epoch={epoch_time:.1f}s"
                      f"  avg={sum(epoch_times)/len(epoch_times):.1f}s")
                print(f"{'─'*70}\n")

                for threshold, msg in [
                    (500, '📉 ADE<500'), (300, '📉 ADE<300'),
                    (200, '🎯 ADE<200'), (150, '🏆 ADE<150'),
                    (100, '🌟 ADE<100!'), (50, '🔥 ADE<50km!!!')
                ]:
                    if ade < threshold:
                        print(f"  {msg} km"); break

                with open(log_path, 'a') as f:
                    f.write(
                        f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f"{ade:.1f},{fde:.1f},"
                        f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
                        f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
                        f"{args.sigma_min:.4f},"
                        f"{epoch_time:.1f},{sms:.1f},"
                        f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
                        f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
                        f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
                        f"{avg_breakdown['ns_cons']:.4f}\n"
                    )

                # Early stopping + best model dựa trên val ADE
                saver(ade, model, args.output_dir, epoch,
                      optimizer, avg_train, avg_val)

            else:
                print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
                      f"  val={avg_val:.4f}  time={epoch_time:.1f}s\n")
                with open(log_path, 'a') as f:
                    f.write(
                        f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f",,,,,,"
                        f"{args.sigma_min:.4f},"
                        f"{epoch_time:.1f},,"
                        f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
                        f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
                        f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
                        f"{avg_breakdown['ns_cons']:.4f}\n"
                    )

            if saver.early_stop:
                print("  Early stopping triggered."); break

        else:
            print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
                  f"  time={epoch_time:.1f}s\n")

        if (epoch + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()}, ckpt)
            print(f"  ✓ Checkpoint → {ckpt}\n")

    # ── Final eval trên test/ (held-out) ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL TEST EVALUATION  (held-out, year={args.test_year})")
    print(f"{'='*70}")

    if test_loader:
        # Load best model
        best_ckpt = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_ckpt):
            ckpt_data = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            print(f"  Loaded best model from epoch {ckpt_data['epoch']}"
                  f"  (val ADE={ckpt_data['val_ade_km']:.1f} km)")

        test_m, _ = evaluate_km(
            model, test_loader, device,
            num_ensemble=args.val_ensemble,
            ddim_steps=args.ode_steps,
            pred_len=args.pred_len,
        )
        print(f"\n  Test (km) │"
              f"  ADE={test_m['ADE']:.1f}  FDE={test_m['FDE']:.1f}"
              f"  12h={test_m.get('12h',0):.0f}"
              f"  24h={test_m.get('24h',0):.0f}"
              f"  48h={test_m.get('48h',0):.0f}"
              f"  72h={test_m.get('72h',0):.0f}")

        # Ghi test results ra file riêng
        test_log = os.path.join(args.output_dir, 'test_results.txt')
        with open(test_log, 'w') as f:
            f.write(f"Test year : {args.test_year}\n")
            f.write(f"ADE (km)  : {test_m['ADE']:.1f}\n")
            f.write(f"FDE (km)  : {test_m['FDE']:.1f}\n")
            f.write(f"12h (km)  : {test_m.get('12h',0):.1f}\n")
            f.write(f"24h (km)  : {test_m.get('24h',0):.1f}\n")
            f.write(f"48h (km)  : {test_m.get('48h',0):.1f}\n")
            f.write(f"72h (km)  : {test_m.get('72h',0):.1f}\n")
        print(f"\n  Test results → {test_log}")
    else:
        print("  ⚠️  Không có test loader, bỏ qua.")

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"  Best val ADE    : {saver.best_ade:.1f} km")
    print(f"  Avg epoch time  : {avg_epoch_time:.1f}s")
    print(f"  Train log       : {log_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    main(args)