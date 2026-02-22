"""
scripts/train_diffusion_v3.py
==============================
Training với ensemble validation và đầy đủ km metrics.
Dùng cùng với diffusion_model.py v4.
"""

import argparse, os, sys, math
import torch, torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TCNM.data.loader import data_loader
from TCNM.diffusion_model import TCDiffusion
from TCNM.utils import get_cosine_schedule_with_warmup


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root',        default='TCND_vn',                type=str)
    p.add_argument('--obs_len',             default=8,                         type=int)
    p.add_argument('--pred_len',            default=12,                        type=int)
    p.add_argument('--num_diffusion_steps', default=100,                       type=int,
                   help='Giảm xuống 100 (từ 250) để train nhanh hơn, DDIM bù lại')
    p.add_argument('--batch_size',          default=32,                        type=int)
    p.add_argument('--num_epochs',          default=300,                       type=int)
    p.add_argument('--g_learning_rate',     default=2e-4,                      type=float)
    p.add_argument('--weight_decay',        default=1e-4,                      type=float)
    p.add_argument('--warmup_epochs',       default=5,                         type=int)
    p.add_argument('--grad_clip',           default=1.0,                       type=float)
    p.add_argument('--patience',            default=40,                        type=int)
    p.add_argument('--gpu_num',             default='0',                       type=str)
    p.add_argument('--output_dir',          default='model_save/diffusion_v3', type=str)
    p.add_argument('--save_interval',       default=10,                        type=int)
    p.add_argument('--val_year',            default=2019,                      type=int)
    p.add_argument('--val_ensemble',        default=5,                         type=int,
                   help='Số samples ensemble khi validate')
    p.add_argument('--ddim_steps',          default=20,                        type=int,
                   help='Bước DDIM sampling (20 là đủ và nhanh)')
    p.add_argument('--val_freq',            default=5,                         type=int,
                   help='Validate km metrics mỗi N epoch')
    # compat
    p.add_argument('--d_model',    default=128, type=int)
    p.add_argument('--delim',      default=' ')
    p.add_argument('--skip',       default=1,   type=int)
    p.add_argument('--min_ped',    default=1,   type=int)
    p.add_argument('--threshold',  default=0.002, type=float)
    p.add_argument('--other_modal',default='gph')
    return p.parse_args()


def resolve_data_path(root):
    root = root.rstrip('/\\')
    if root.endswith(('Data1d/train','Data1d\\train')):
        return root, root[:-5] + 'test'
    if root.endswith(('Data1d/test','Data1d\\test')):
        return root[:-4] + 'train', root
    if root.endswith('Data1d'):
        return os.path.join(root,'train'), os.path.join(root,'test')
    return os.path.join(root,'Data1d','train'), os.path.join(root,'Data1d','test')


def move_batch(bl, device):
    for j,x in enumerate(bl):
        if torch.is_tensor(x):
            bl[j] = x.to(device)
        elif isinstance(x, dict):
            bl[j] = {k: v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
    return bl


def denorm_traj(n):
    r = n.clone()
    r[...,0] = n[...,0]*50+1800
    r[...,1] = n[...,1]*50
    return r


def validate_km(model, loader, device, num_ensemble=5, ddim_steps=20, pred_len=12):
    """Tính ADE/FDE/24h/48h/72h error bằng km dùng ensemble sampling."""
    model.eval()
    all_step_errors = []

    with torch.no_grad():
        for batch in loader:
            bl  = move_batch(list(batch), device)
            gt  = bl[1]   # [T, B, 2]

            # Ensemble sample
            pred, _ = model.sample(bl,
                                   num_ensemble=num_ensemble,
                                   ddim_steps=ddim_steps)  # [T, B, 2]

            pred_r = denorm_traj(pred)
            gt_r   = denorm_traj(gt)

            dist = torch.norm(pred_r - gt_r, dim=2) * 11.1  # [T, B] km
            all_step_errors.append(dist.mean(dim=1).cpu())   # [T]

    stacked = torch.stack(all_step_errors, dim=0)  # [n_batches, T]
    mean_steps = stacked.mean(dim=0)               # [T]

    m = {
        'ADE': mean_steps.mean().item(),
        'FDE': mean_steps[-1].item(),
    }
    for h, s in [(12,1),(24,3),(48,7),(72,11)]:
        if s < pred_len:
            m[f'{h}h'] = mean_steps[s].item()

    return m, mean_steps


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
                'args': {'obs_len': model.obs_len, 'pred_len': model.pred_len},
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


def main(args):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*65)
    print("  TC-DIFFUSION v4  (Ensemble DDIM + Regression Head)")
    print("="*65)
    print(f"  Device       : {device}")
    print(f"  Diffusion T  : {args.num_diffusion_steps}  DDIM steps: {args.ddim_steps}")
    print(f"  Val ensemble : {args.val_ensemble}")
    print("="*65)

    train_dir, test_dir = resolve_data_path(args.dataset_root)

    _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)
    val_loader = None
    if os.path.exists(test_dir):
        _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                    test=True, test_year=args.val_year)

    print(f"  Train: {len(train_loader.dataset)} seq  "
          f"Val: {len(val_loader.dataset) if val_loader else 0} seq\n")

    model = TCDiffusion(
        pred_len  = args.pred_len,
        obs_len   = args.obs_len,
        num_steps = args.num_diffusion_steps,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_p:,}\n")

    optimizer = optim.AdamW(model.parameters(),
                             lr=args.g_learning_rate,
                             weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    saver = BestModelSaver(patience=args.patience, verbose=True)

    log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,ADE_km,FDE_km,12h,24h,48h,72h,blend_w\n")

    print("="*65 + "\n  TRAINING\n" + "="*65)

    for epoch in range(args.num_epochs):

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            bl   = move_batch(list(batch), device)
            loss = model.get_loss(bl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if i % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                bw = torch.sigmoid(model.denoiser.blend_logit).item()
                print(f"  [{epoch:>3}/{args.num_epochs}] [{i:>3}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}  lr={lr:.2e}  blend={bw:.2f}")

        avg_train = train_loss / len(train_loader)
        blend_w   = torch.sigmoid(model.denoiser.blend_logit).item()

        # ── Validate ──────────────────────────────────────────────────
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    bl = move_batch(list(batch), device)
                    val_loss += model.get_loss(bl).item()
            avg_val = val_loss / len(val_loader)

            # km metrics mỗi val_freq epoch
            if epoch % args.val_freq == 0 or epoch < 5:
                m, per_step = validate_km(
                    model, val_loader, device,
                    num_ensemble=args.val_ensemble,
                    ddim_steps=args.ddim_steps,
                    pred_len=args.pred_len,
                )
                ade = m['ADE']
                fde = m['FDE']

                print(f"\n{'─'*65}")
                print(f"  Epoch {epoch:>3}  │  train={avg_train:.4f}  val={avg_val:.4f}  blend={blend_w:.2f}")
                print(f"  Track (km)  │  ADE={ade:.1f}  FDE={fde:.1f}  "
                      f"12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}  "
                      f"48h={m.get('48h',0):.0f}  72h={m.get('72h',0):.0f}")
                print(f"{'─'*65}\n")

                # Milestones
                for threshold, msg in [(500,'📉 ADE<500'),(300,'📉 ADE<300'),
                                        (200,'🎯 ADE<200'),(150,'🏆 ADE<150'),
                                        (100,'🌟 ADE<100!'),(50,'🔥 ADE<50km!!!')]:
                    if ade < threshold:
                        print(f"  {msg} km")
                        break

                with open(log_path, 'a') as f:
                    f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                            f"{ade:.1f},{fde:.1f},"
                            f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
                            f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
                            f"{blend_w:.3f}\n")

                saver(ade, model, args.output_dir, epoch, optimizer, avg_train, avg_val)
            else:
                print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}  "
                      f"val={avg_val:.4f}  blend={blend_w:.2f}\n")

                with open(log_path, 'a') as f:
                    f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},,,,,,,{blend_w:.3f}\n")

            if saver.early_stop:
                print("  Early stopping."); break
        else:
            print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}\n")

        if (epoch + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt)
            print(f"  ✓ Checkpoint → {ckpt}\n")

    print(f"\n{'='*65}")
    print(f"  DONE  Best ADE: {saver.best_ade:.1f} km")
    print(f"  Log: {log_path}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    main(args)

# """
# scripts/train_diffusion_v3.py
# ==============================
# Training với ensemble validation và đầy đủ km metrics.
# Dùng cùng với diffusion_model.py v4.
# """

# import argparse, os, sys, math
# import torch, torch.optim as optim
# import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from TCNM.data.loader import data_loader
# from TCNM.diffusion_model import TCDiffusion
# from TCNM.utils import get_cosine_schedule_with_warmup


# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--dataset_root',        default='TCND_vn',                type=str)
#     p.add_argument('--obs_len',             default=8,                         type=int)
#     p.add_argument('--pred_len',            default=12,                        type=int)
#     p.add_argument('--num_diffusion_steps', default=100,                       type=int,
#                    help='Giảm xuống 100 (từ 250) để train nhanh hơn, DDIM bù lại')
#     p.add_argument('--batch_size',          default=32,                        type=int)
#     p.add_argument('--num_epochs',          default=300,                       type=int)
#     p.add_argument('--g_learning_rate',     default=2e-4,                      type=float)
#     p.add_argument('--weight_decay',        default=1e-4,                      type=float)
#     p.add_argument('--warmup_epochs',       default=5,                         type=int)
#     p.add_argument('--grad_clip',           default=1.0,                       type=float)
#     p.add_argument('--patience',            default=40,                        type=int)
#     p.add_argument('--gpu_num',             default='0',                       type=str)
#     p.add_argument('--output_dir',          default='model_save/diffusion_v3', type=str)
#     p.add_argument('--save_interval',       default=10,                        type=int)
#     p.add_argument('--val_year',            default=2019,                      type=int)
#     p.add_argument('--val_ensemble',        default=5,                         type=int,
#                    help='Số samples ensemble khi validate')
#     p.add_argument('--ddim_steps',          default=20,                        type=int,
#                    help='Bước DDIM sampling (20 là đủ và nhanh)')
#     p.add_argument('--val_freq',            default=5,                         type=int,
#                    help='Validate km metrics mỗi N epoch')
#     # compat
#     p.add_argument('--d_model',    default=128, type=int)
#     p.add_argument('--delim',      default=' ')
#     p.add_argument('--skip',       default=1,   type=int)
#     p.add_argument('--min_ped',    default=1,   type=int)
#     p.add_argument('--threshold',  default=0.002, type=float)
#     p.add_argument('--other_modal',default='gph')
#     return p.parse_args()


# def resolve_data_path(root):
#     """
#     Ưu tiên dùng val/ nếu có, fallback sang test/.
#     Hỗ trợ cấu trúc: TCND_vn/Data1d/{train, val, test}
#     """
#     root = root.rstrip('/\\')

#     # Trỏ thẳng vào .../Data1d/train
#     if root.endswith('Data1d/train') or root.endswith('Data1d\\train'):
#         base = root[:-len('train')]
#         val  = base + 'val'
#         test = base + 'test'
#         return root, val if os.path.exists(val) else test

#     # Trỏ vào .../Data1d
#     if os.path.basename(root) == 'Data1d':
#         val  = os.path.join(root, 'val')
#         test = os.path.join(root, 'test')
#         return os.path.join(root, 'train'), val if os.path.exists(val) else test

#     # Mặc định: root = TCND_vn
#     data1d = os.path.join(root, 'Data1d')
#     val    = os.path.join(data1d, 'val')
#     test   = os.path.join(data1d, 'test')
#     return (os.path.join(data1d, 'train'),
#             val if os.path.exists(val) else test)


# def move_batch(bl, device):
#     for j,x in enumerate(bl):
#         if torch.is_tensor(x):
#             bl[j] = x.to(device)
#         elif isinstance(x, dict):
#             bl[j] = {k: v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
#     return bl


# def denorm_traj(n):
#     r = n.clone()
#     r[...,0] = n[...,0]*50+1800
#     r[...,1] = n[...,1]*50
#     return r


# def validate_km(model, loader, device, num_ensemble=5, ddim_steps=20, pred_len=12):
#     """Tính ADE/FDE/24h/48h/72h error bằng km dùng ensemble sampling."""
#     model.eval()
#     all_step_errors = []

#     with torch.no_grad():
#         for batch in loader:
#             bl  = move_batch(list(batch), device)
#             gt  = bl[1]   # [T, B, 2]

#             # Ensemble sample
#             pred, _ = model.sample(bl,
#                                    num_ensemble=num_ensemble,
#                                    ddim_steps=ddim_steps)  # [T, B, 2]

#             pred_r = denorm_traj(pred)
#             gt_r   = denorm_traj(gt)

#             dist = torch.norm(pred_r - gt_r, dim=2) * 11.1  # [T, B] km
#             all_step_errors.append(dist.mean(dim=1).cpu())   # [T]

#     stacked = torch.stack(all_step_errors, dim=0)  # [n_batches, T]
#     mean_steps = stacked.mean(dim=0)               # [T]

#     m = {
#         'ADE': mean_steps.mean().item(),
#         'FDE': mean_steps[-1].item(),
#     }
#     for h, s in [(12,1),(24,3),(48,7),(72,11)]:
#         if s < pred_len:
#             m[f'{h}h'] = mean_steps[s].item()

#     return m, mean_steps


# class BestModelSaver:
#     def __init__(self, patience=40, min_delta=2.0, verbose=True):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.verbose    = verbose
#         self.counter    = 0
#         self.best_ade   = float('inf')
#         self.early_stop = False

#     def __call__(self, ade_km, model, out_dir, epoch, opt, train_loss, val_loss):
#         if ade_km < self.best_ade - self.min_delta:
#             self.best_ade = ade_km
#             self.counter  = 0
#             ckpt = os.path.join(out_dir, 'best_model.pth')
#             torch.save({
#                 'epoch':            epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state':  opt.state_dict(),
#                 'train_loss':       train_loss,
#                 'val_loss':         val_loss,
#                 'val_ade_km':       ade_km,
#                 'args': {'obs_len': model.obs_len, 'pred_len': model.pred_len},
#             }, ckpt)
#             if self.verbose:
#                 print(f"  ✅ Best ADE: {ade_km:.1f} km  →  saved {ckpt}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"  EarlyStopping: {self.counter}/{self.patience}  "
#                       f"(best={self.best_ade:.1f} km)")
#             if self.counter >= self.patience:
#                 self.early_stop = True


# def main(args):
#     if torch.cuda.is_available():
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     os.makedirs(args.output_dir, exist_ok=True)

#     print("="*65)
#     print("  TC-DIFFUSION v4  (Ensemble DDIM + Regression Head)")
#     print("="*65)
#     print(f"  Device       : {device}")
#     print(f"  Diffusion T  : {args.num_diffusion_steps}  DDIM steps: {args.ddim_steps}")
#     print(f"  Val ensemble : {args.val_ensemble}")
#     print("="*65)

#     train_dir, test_dir = resolve_data_path(args.dataset_root)

#     _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)
#     val_loader = None
#     if os.path.exists(test_dir):
#         val_type = 'val' if 'val' in os.path.basename(test_dir) else 'test'
#         # Nếu dùng thư mục val/ riêng thì không cần filter theo năm
#         use_year = None if val_type == 'val' else args.val_year
#         _, val_loader = data_loader(args, {'root': test_dir, 'type': val_type},
#                                     test=True, test_year=use_year)
#         print(f"  Val dir: {test_dir}  (type={val_type}, year_filter={use_year})")

#     print(f"  Train: {len(train_loader.dataset)} seq  "
#           f"Val: {len(val_loader.dataset) if val_loader else 0} seq\n")

#     model = TCDiffusion(
#         pred_len  = args.pred_len,
#         obs_len   = args.obs_len,
#         num_steps = args.num_diffusion_steps,
#     ).to(device)

#     n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  Parameters: {n_p:,}\n")

#     optimizer = optim.AdamW(model.parameters(),
#                              lr=args.g_learning_rate,
#                              weight_decay=args.weight_decay)
#     total_steps  = len(train_loader) * args.num_epochs
#     warmup_steps = len(train_loader) * args.warmup_epochs
#     scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

#     saver = BestModelSaver(patience=args.patience, verbose=True)

#     log_path = os.path.join(args.output_dir, 'training_log.csv')
#     with open(log_path, 'w') as f:
#         f.write("epoch,train_loss,val_loss,ADE_km,FDE_km,12h,24h,48h,72h,blend_w\n")

#     print("="*65 + "\n  TRAINING\n" + "="*65)

#     for epoch in range(args.num_epochs):

#         # ── Train ─────────────────────────────────────────────────────
#         model.train()
#         train_loss = 0.0
#         for i, batch in enumerate(train_loader):
#             bl   = move_batch(list(batch), device)
#             loss = model.get_loss(bl)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()
#             scheduler.step()
#             train_loss += loss.item()

#             if i % 20 == 0:
#                 lr = optimizer.param_groups[0]['lr']
#                 bw = torch.sigmoid(model.denoiser.blend_logit).item()
#                 print(f"  [{epoch:>3}/{args.num_epochs}] [{i:>3}/{len(train_loader)}] "
#                       f"loss={loss.item():.4f}  lr={lr:.2e}  blend={bw:.2f}")

#         avg_train = train_loss / len(train_loader)
#         blend_w   = torch.sigmoid(model.denoiser.blend_logit).item()

#         # ── Validate ──────────────────────────────────────────────────
#         if val_loader:
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     bl = move_batch(list(batch), device)
#                     val_loss += model.get_loss(bl).item()
#             avg_val = val_loss / len(val_loader)

#             # km metrics mỗi val_freq epoch
#             if epoch % args.val_freq == 0 or epoch < 5:
#                 m, per_step = validate_km(
#                     model, val_loader, device,
#                     num_ensemble=args.val_ensemble,
#                     ddim_steps=args.ddim_steps,
#                     pred_len=args.pred_len,
#                 )
#                 ade = m['ADE']
#                 fde = m['FDE']

#                 print(f"\n{'─'*65}")
#                 print(f"  Epoch {epoch:>3}  │  train={avg_train:.4f}  val={avg_val:.4f}  blend={blend_w:.2f}")
#                 print(f"  Track (km)  │  ADE={ade:.1f}  FDE={fde:.1f}  "
#                       f"12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}  "
#                       f"48h={m.get('48h',0):.0f}  72h={m.get('72h',0):.0f}")
#                 print(f"{'─'*65}\n")

#                 # Milestones
#                 for threshold, msg in [(500,'📉 ADE<500'),(300,'📉 ADE<300'),
#                                         (200,'🎯 ADE<200'),(150,'🏆 ADE<150'),
#                                         (100,'🌟 ADE<100!'),(50,'🔥 ADE<50km!!!')]:
#                     if ade < threshold:
#                         print(f"  {msg} km")
#                         break

#                 with open(log_path, 'a') as f:
#                     f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},"
#                             f"{ade:.1f},{fde:.1f},"
#                             f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
#                             f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
#                             f"{blend_w:.3f}\n")

#                 saver(ade, model, args.output_dir, epoch, optimizer, avg_train, avg_val)
#             else:
#                 print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}  "
#                       f"val={avg_val:.4f}  blend={blend_w:.2f}\n")

#                 with open(log_path, 'a') as f:
#                     f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},,,,,,,{blend_w:.3f}\n")

#             if saver.early_stop:
#                 print("  Early stopping."); break
#         else:
#             print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}\n")

#         if (epoch + 1) % args.save_interval == 0:
#             ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
#             torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt)
#             print(f"  ✓ Checkpoint → {ckpt}\n")

#     print(f"\n{'='*65}")
#     print(f"  DONE  Best ADE: {saver.best_ade:.1f} km")
#     print(f"  Log: {log_path}")
#     print(f"{'='*65}\n")


# if __name__ == '__main__':
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(42)
#     main(args)