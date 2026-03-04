"""
scripts/train_flowmatching.py
==============================
Training TCFlowMatching v3 với logging chi tiết để so sánh với Diffusion.

So sánh metrics:
  - ADE/FDE/24h/48h/72h (km) — giống diffusion
  - Train time per epoch (s)
  - Sample time per batch (ms)
  - Loss breakdown: fm_loss, dir_loss, smooth_loss, disp_loss, curv_loss, ns_loss
  - blend_w = 0.5 (không có, hiển thị N/A)

Chạy:
  python scripts/train_flowmatching.py \
      --dataset_root TCND_vn \
      --output_dir   model_save/flowmatching_v3 \
      --ddim_steps 10 --sigma_min 0.001 \
      --num_epochs 200 --batch_size 32
"""

import argparse, os, sys, time, math
import torch, torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TCNM.data.loader import data_loader
from TCNM.flow_matching_model import TCFlowMatching
from TCNM.utils import get_cosine_schedule_with_warmup


# ── Helper: không crash nếu model không có blend_logit ───────────────────────
def get_blend_w(model):
    if hasattr(model, 'denoiser') and hasattr(model.denoiser, 'blend_logit'):
        return torch.sigmoid(model.denoiser.blend_logit).item()
    if hasattr(model, 'net') and hasattr(model.net, 'blend_logit'):
        return torch.sigmoid(model.net.blend_logit).item()
    if hasattr(model, 'blend_logit'):
        return torch.sigmoid(model.blend_logit).item()
    return -1.0   # sentinel: N/A


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root',        default='TCND_vn',                  type=str)
    p.add_argument('--obs_len',             default=8,                           type=int)
    p.add_argument('--pred_len',            default=12,                          type=int)
    p.add_argument('--num_diffusion_steps', default=100,                         type=int)
    p.add_argument('--batch_size',          default=32,                          type=int)
    p.add_argument('--num_epochs',          default=200,                         type=int)
    p.add_argument('--g_learning_rate',     default=2e-4,                        type=float)
    p.add_argument('--weight_decay',        default=1e-4,                        type=float)
    p.add_argument('--warmup_epochs',       default=5,                           type=int)
    p.add_argument('--grad_clip',           default=1.0,                         type=float)
    p.add_argument('--patience',            default=40,                          type=int)
    p.add_argument('--gpu_num',             default='0',                         type=str)
    p.add_argument('--output_dir',          default='model_save/flowmatching_v3',type=str)
    p.add_argument('--save_interval',       default=10,                          type=int)
    p.add_argument('--val_year',            default=2019,                        type=int)
    p.add_argument('--val_ensemble',        default=5,                           type=int)
    p.add_argument('--ddim_steps',          default=10,                          type=int,
                   help='ODE steps — OT-CFM chỉ cần 10 (Diffusion dùng 20)')
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
    root = root.rstrip('/\\')
    if root.endswith(('Data1d/train', 'Data1d\\train')):
        return root, root[:-5] + 'test'
    if root.endswith(('Data1d/test',  'Data1d\\test')):
        return root[:-4] + 'train', root
    if root.endswith('Data1d'):
        return os.path.join(root, 'train'), os.path.join(root, 'test')
    return os.path.join(root, 'Data1d', 'train'), os.path.join(root, 'Data1d', 'test')


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


# ── Loss breakdown (v4 — PINN vorticity) ─────────────────────────────────────
def compute_loss_breakdown(model, batch_list):
    """
    Breakdown loss để log từng thành phần.
    Compatible với TCFlowMatching v4 (OT-CFM + PINN vorticity).
    """
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

    # OT-CFM interpolation + target
    x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
    denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
    target_vel = (x1 - (1 - sm) * x_t) / denom

    # v4: dùng model.net.forward() bình thường (không có forward_with_ns)
    pred_vel    = model.net(x_t, t, batch_list)
    fm_loss     = F.mse_loss(pred_vel, target_vel)

    pred_x1     = x_t + denom * pred_vel
    pred_abs, _ = model.rel_to_abs(pred_x1, lp, lm)

    dir_l  = model._dir_loss(pred_abs, traj_gt, lp)
    smt_l  = model._smooth_loss(pred_x1)
    disp_l = model._weighted_disp_loss(pred_abs, traj_gt)
    curv_l = model._curvature_loss(pred_abs, traj_gt)

    # PINN: vorticity equation residual (thay ns_loss của v3)
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
        'ns':      pinn_l.item(),   # key 'ns' giữ để compat log CSV
        'ns_cons': 0.0,
    }


# ── Validation ────────────────────────────────────────────────────────────────
def validate_km(model, loader, device, num_ensemble=5,
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
                'model_type':       'TCFlowMatching_v3_OT-CFM+NS',
                'args': {
                    'obs_len':   model.obs_len,
                    'pred_len':  model.pred_len,
                    'sigma_min': model.sigma_min,
                },
            }, ckpt)
            if self.verbose:
                print(f"  ✅ Best ADE: {ade_km:.1f} km  →  saved {ckpt}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}  "
                      f"(best={self.best_ade:.1f} km)")
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
    print("  TC-FlowMatching v3  |  OT-CFM + Navier-Stokes Physics Prior")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  ODE steps   : {args.ddim_steps}  "
          f"(Diffusion dùng 20 DDIM steps → FM nhanh ~2x sampling)")
    print(f"  sigma_min   : {args.sigma_min}  (OT path, near-deterministic)")
    print(f"  Ensemble    : {args.val_ensemble}")
    print(f"  LR          : {args.g_learning_rate}  WD: {args.weight_decay}")
    print(f"  Epochs      : {args.num_epochs}  Patience: {args.patience}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    train_dir, test_dir = resolve_data_path(args.dataset_root)
    _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)
    val_loader = None
    if os.path.exists(test_dir):
        _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                    test=True, test_year=args.val_year)

    print(f"  Train: {len(train_loader.dataset)} seq  "
          f"Val: {len(val_loader.dataset) if val_loader else 0} seq\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len  = args.pred_len,
        obs_len   = args.obs_len,
        num_steps = args.num_diffusion_steps,
        sigma_min = args.sigma_min,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  : {n_p:,}")
    print(f"  Architecture: VelocityField + NavierStokesPhysics\n")

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
        # Cùng format với diffusion log + thêm cột cho FM
        f.write("epoch,train_loss,val_loss,ADE_km,FDE_km,"
                "12h,24h,48h,72h,blend_w,"
                "epoch_time_s,sample_ms_per_batch,"
                "fm_loss,dir_loss,smooth_loss,disp_loss,curv_loss,ns_loss,ns_cons\n")

    # Comparison header
    comp_path = os.path.join(args.output_dir, 'comparison_vs_diffusion.txt')
    with open(comp_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  FlowMatching v3 vs Diffusion v5 — Comparison Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"  FM sigma_min  : {args.sigma_min}\n")
        f.write(f"  FM ODE steps  : {args.ddim_steps}  (Diff: 20 DDIM)\n")
        f.write(f"  FM ensemble   : {args.val_ensemble}\n")
        f.write(f"  FM physics    : OT-CFM + Navier-Stokes\n")
        f.write(f"  Diff physics  : DDIM + DirectRegression + blend\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Epoch':>6} {'FM_ADE':>8} {'Diff_ADE':>10} "
                f"{'FM_72h':>8} {'Diff_72h':>10} "
                f"{'EpochTime':>10} {'SampleMs':>10}\n")
        f.write("-" * 60 + "\n")

    # Diffusion best ADE reference (từ training trước)
    DIFF_BEST_ADE = 353.4   # epoch 80 từ training log trước
    DIFF_72H      = None    # điền vào nếu có

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
            bl     = move_batch(list(batch), device)
            bd     = compute_loss_breakdown(model, bl)
            loss   = bd['total']

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
        blend_w       = get_blend_w(model)
        blend_str     = f"{blend_w:.3f}" if blend_w >= 0 else "N/A"

        # ── Validate ───────────────────────────────────────────────────
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    bl = move_batch(list(batch), device)
                    val_loss += model.get_loss(bl).item()
            avg_val = val_loss / len(val_loader)

            if epoch % args.val_freq == 0 or epoch < 5:
                m, per_step = validate_km(
                    model, val_loader, device,
                    num_ensemble=args.val_ensemble,
                    ddim_steps=args.ddim_steps,
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
                print(f"  Track (km)     │"
                      f"  ADE={ade:.1f}  FDE={fde:.1f}"
                      f"  12h={m.get('12h',0):.0f}"
                      f"  24h={m.get('24h',0):.0f}"
                      f"  48h={m.get('48h',0):.0f}"
                      f"  72h={m.get('72h',0):.0f}")
                print(f"  Speed          │"
                      f"  sample={sms:.1f}ms/batch"
                      f"  epoch={epoch_time:.1f}s"
                      f"  avg={sum(epoch_times)/len(epoch_times):.1f}s")

                # So sánh với diffusion
                delta_ade = ade - DIFF_BEST_ADE
                sign      = "▲" if delta_ade > 0 else "▼"
                print(f"  vs Diffusion   │"
                      f"  Diff_best={DIFF_BEST_ADE:.1f}km"
                      f"  FM={ade:.1f}km"
                      f"  {sign}{abs(delta_ade):.1f}km"
                      f"  ({'worse' if delta_ade > 0 else 'BETTER ✅'})")
                print(f"{'─'*70}\n")

                # Milestones
                for threshold, msg in [
                    (500, '📉 ADE<500'), (300, '📉 ADE<300'),
                    (200, '🎯 ADE<200'), (150, '🏆 ADE<150'),
                    (100, '🌟 ADE<100!'), (50,  '🔥 ADE<50km!!!')
                ]:
                    if ade < threshold:
                        print(f"  {msg} km"); break

                # CSV log (compatible với diffusion log format)
                with open(log_path, 'a') as f:
                    f.write(
                        f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f"{ade:.1f},{fde:.1f},"
                        f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
                        f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
                        f"{blend_str},"
                        f"{epoch_time:.1f},{sms:.1f},"
                        f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
                        f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
                        f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
                        f"{avg_breakdown['ns_cons']:.4f}\n"
                    )

                # Comparison log
                with open(comp_path, 'a') as f:
                    f.write(
                        f"{epoch:>6} {ade:>8.1f} {DIFF_BEST_ADE:>10.1f} "
                        f"{m.get('72h',0):>8.0f} {'N/A':>10} "
                        f"{epoch_time:>10.1f} {sms:>10.1f}\n"
                    )

                saver(ade, model, args.output_dir, epoch,
                      optimizer, avg_train, avg_val)

            else:
                print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
                      f"  val={avg_val:.4f}  time={epoch_time:.1f}s\n")
                with open(log_path, 'a') as f:
                    f.write(
                        f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f",,,,,,"
                        f"{blend_str},"
                        f"{epoch_time:.1f},,"
                        f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
                        f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
                        f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
                        f"{avg_breakdown['ns_cons']:.4f}\n"
                    )

            if saver.early_stop:
                print("  Early stopping."); break

        else:
            print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
                  f"  time={epoch_time:.1f}s\n")

        if (epoch + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()}, ckpt)
            print(f"  ✓ Checkpoint → {ckpt}\n")

    # ── Final summary ─────────────────────────────────────────────────
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"  Best FM ADE     : {saver.best_ade:.1f} km")
    print(f"  Diff best ADE   : {DIFF_BEST_ADE:.1f} km  (epoch 80)")
    delta = saver.best_ade - DIFF_BEST_ADE
    print(f"  Delta           : {'▲' if delta > 0 else '▼'}{abs(delta):.1f} km"
          f"  ({'FM worse' if delta > 0 else 'FM BETTER ✅'})")
    print(f"  Avg epoch time  : {avg_epoch_time:.1f}s")
    print(f"  Log             : {log_path}")
    print(f"  Comparison      : {comp_path}")
    print(f"{'='*70}\n")

    # Final comparison file summary
    with open(comp_path, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"  SUMMARY\n")
        f.write(f"  FM best ADE     : {saver.best_ade:.1f} km\n")
        f.write(f"  Diffusion ADE   : {DIFF_BEST_ADE:.1f} km\n")
        f.write(f"  Delta           : {delta:+.1f} km\n")
        f.write(f"  Avg epoch time  : {avg_epoch_time:.1f}s\n")
        f.write("=" * 60 + "\n")


if __name__ == '__main__':
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    main(args)