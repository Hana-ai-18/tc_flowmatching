# # """
# # scripts/train_flowmatching.py
# # ==============================
# # Training TCFlowMatching v4 với OT-CFM + PINN Vorticity.

# # Metrics:
# #   - ADE/FDE/24h/48h/72h (km)
# #   - Train time per epoch (s)
# #   - Sample time per batch (ms)
# #   - Loss breakdown: fm, dir, smooth, disp, curv, pinn(NS)

# # Data split:
# #   - train/  : training data
# #   - val/    : validation (early stopping, best model selection)
# #   - test/   : held-out test set (chỉ đánh giá cuối cùng, KHÔNG dùng để tune)

# # Chạy:
# #   python scripts/train_flowmatching.py \
# #       --dataset_root TCND_vn \
# #       --output_dir   model_save/flowmatching_v4 \
# #       --ode_steps 10 --sigma_min 0.001 \
# #       --num_epochs 200 --batch_size 32
# # """

# # import argparse, os, sys, time, math
# # import torch, torch.optim as optim
# # import torch.nn.functional as F
# # import numpy as np

# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # from TCNM.data.loader import data_loader
# # from TCNM.flow_matching_model import TCFlowMatching
# # from TCNM.utils import get_cosine_schedule_with_warmup


# # def get_args():
# #     p = argparse.ArgumentParser()
# #     p.add_argument('--dataset_root',        default='TCND_vn',                  type=str)
# #     p.add_argument('--obs_len',             default=8,                           type=int)
# #     p.add_argument('--pred_len',            default=12,                          type=int)
# #     p.add_argument('--batch_size',          default=32,                          type=int)
# #     p.add_argument('--num_epochs',          default=200,                         type=int)
# #     p.add_argument('--g_learning_rate',     default=2e-4,                        type=float)
# #     p.add_argument('--weight_decay',        default=1e-4,                        type=float)
# #     p.add_argument('--warmup_epochs',       default=5,                           type=int)
# #     p.add_argument('--grad_clip',           default=1.0,                         type=float)
# #     p.add_argument('--patience',            default=40,                          type=int)
# #     p.add_argument('--gpu_num',             default='0',                         type=str)
# #     p.add_argument('--output_dir',          default='model_save/flowmatching_v4',type=str)
# #     p.add_argument('--save_interval',       default=10,                          type=int)
# #     p.add_argument('--test_year',           default=2019,                        type=int,
# #                    help='Năm để filter test set (held-out, chỉ eval cuối)')
# #     p.add_argument('--val_ensemble',        default=5,                           type=int)
# #     p.add_argument('--ode_steps',           default=10,                          type=int,
# #                    help='ODE integration steps cho Flow Matching')
# #     p.add_argument('--val_freq',            default=2,                           type=int)
# #     p.add_argument('--sigma_min',           default=0.001,                       type=float)
# #     # compat
# #     p.add_argument('--d_model',    default=128,   type=int)
# #     p.add_argument('--delim',      default=' ')
# #     p.add_argument('--skip',       default=1,     type=int)
# #     p.add_argument('--min_ped',    default=1,     type=int)
# #     p.add_argument('--threshold',  default=0.002, type=float)
# #     p.add_argument('--other_modal',default='gph')
# #     return p.parse_args()


# # def resolve_data_path(root):
# #     """
# #     Trả về (train_dir, val_dir, test_dir).

# #     Ưu tiên dùng val/ folder riêng biệt.
# #     Nếu không có val/, fallback về test/ (với cảnh báo).
# #     """
# #     root = root.rstrip('/\\')

# #     # Nếu truyền vào thẳng Data1d/train hoặc tương tự
# #     if root.endswith(('Data1d/train', 'Data1d\\train')):
# #         base = root[:-len('train')]
# #     elif root.endswith(('Data1d/test', 'Data1d\\test')):
# #         base = root[:-len('test')]
# #     elif root.endswith('Data1d'):
# #         base = root + os.sep
# #     else:
# #         base = os.path.join(root, 'Data1d') + os.sep

# #     train_dir = os.path.join(base, 'train')
# #     val_dir   = os.path.join(base, 'val')
# #     test_dir  = os.path.join(base, 'test')

# #     return train_dir, val_dir, test_dir


# # def move_batch(bl, device):
# #     for j, x in enumerate(bl):
# #         if torch.is_tensor(x):
# #             bl[j] = x.to(device)
# #         elif isinstance(x, dict):
# #             bl[j] = {k: v.to(device) if torch.is_tensor(v) else v
# #                      for k, v in x.items()}
# #     return bl


# # def denorm_traj(n):
# #     r = n.clone()
# #     r[..., 0] = n[..., 0] * 50 + 1800
# #     r[..., 1] = n[..., 1] * 50
# #     return r


# # # ── Loss breakdown ────────────────────────────────────────────────────────────
# # def compute_loss_breakdown(model, batch_list):
# #     traj_gt = batch_list[1]
# #     Me_gt   = batch_list[8]
# #     obs     = batch_list[0]
# #     obs_Me  = batch_list[7]

# #     B      = traj_gt.shape[1]
# #     device = traj_gt.device
# #     lp, lm = obs[-1], obs_Me[-1]
# #     sm     = model.sigma_min

# #     x1    = model.traj_to_rel(traj_gt, Me_gt, lp, lm)
# #     x0    = torch.randn_like(x1) * sm
# #     t     = torch.rand(B, device=device)
# #     t_exp = t.view(B, 1, 1)

# #     x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
# #     denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
# #     target_vel = (x1 - (1 - sm) * x_t) / denom

# #     pred_vel    = model.net(x_t, t, batch_list)
# #     fm_loss     = F.mse_loss(pred_vel, target_vel)

# #     pred_x1     = x_t + denom * pred_vel
# #     pred_abs, _ = model.rel_to_abs(pred_x1, lp, lm)

# #     dir_l  = model._dir_loss(pred_abs, traj_gt, lp)
# #     smt_l  = model._smooth_loss(pred_x1)
# #     disp_l = model._weighted_disp_loss(pred_abs, traj_gt)
# #     curv_l = model._curvature_loss(pred_abs, traj_gt)
# #     # Trong compute_loss_breakdown, sau khi tính curv_l:
    
# #     pinn_l = model._ns_pinn_loss(pred_abs)

# #     total = (fm_loss + 2.0*dir_l + 0.5*smt_l
# #              + 1.0*disp_l + 1.5*curv_l + 0.5*pinn_l)

# #     return {
# #         'total':   total,
# #         'fm':      fm_loss.item(),
# #         'dir':     dir_l.item(),
# #         'smooth':  smt_l.item(),
# #         'disp':    disp_l.item(),
# #         'curv':    curv_l.item(),
# #         'ns':      pinn_l.item(),
# #         'ns_cons': 0.0,
# #     }


# # # ── Validation / Test metrics ─────────────────────────────────────────────────
# # def evaluate_km(model, loader, device, num_ensemble=5,
# #                 ddim_steps=10, pred_len=12):
# #     model.eval()
# #     all_step_errors = []
# #     total_sample_ms = 0.0
# #     n_batches       = 0

# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move_batch(list(batch), device)
# #             gt = bl[1]

# #             t0 = time.time()
# #             pred, _ = model.sample(bl, num_ensemble=num_ensemble,
# #                                    ddim_steps=ddim_steps)
# #             total_sample_ms += (time.time() - t0) * 1000
# #             n_batches += 1

# #             pred_r = denorm_traj(pred)
# #             gt_r   = denorm_traj(gt)
# #             dist   = torch.norm(pred_r - gt_r, dim=2) * 11.1
# #             all_step_errors.append(dist.mean(dim=1).cpu())

# #     stacked    = torch.stack(all_step_errors, dim=0)
# #     mean_steps = stacked.mean(dim=0)

# #     m = {'ADE': mean_steps.mean().item(), 'FDE': mean_steps[-1].item()}
# #     for h, s in [(12, 1), (24, 3), (48, 7), (72, 11)]:
# #         if s < pred_len:
# #             m[f'{h}h'] = mean_steps[s].item()

# #     m['sample_ms_per_batch'] = total_sample_ms / max(n_batches, 1)
# #     return m, mean_steps


# # # ── Best Model Saver ──────────────────────────────────────────────────────────
# # class BestModelSaver:
# #     def __init__(self, patience=40, min_delta=2.0, verbose=True):
# #         self.patience   = patience
# #         self.min_delta  = min_delta
# #         self.verbose    = verbose
# #         self.counter    = 0
# #         self.best_ade   = float('inf')
# #         self.early_stop = False

# #     def __call__(self, ade_km, model, out_dir, epoch, opt, train_loss, val_loss):
# #         if ade_km < self.best_ade - self.min_delta:
# #             self.best_ade = ade_km
# #             self.counter  = 0
# #             ckpt = os.path.join(out_dir, 'best_model.pth')
# #             torch.save({
# #                 'epoch':            epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state':  opt.state_dict(),
# #                 'train_loss':       train_loss,
# #                 'val_loss':         val_loss,
# #                 'val_ade_km':       ade_km,
# #                 'model_type':       'TCFlowMatching_v4_OT-CFM+PINN',
# #                 'args': {
# #                     'obs_len':   model.obs_len,
# #                     'pred_len':  model.pred_len,
# #                     'sigma_min': model.sigma_min,
# #                 },
# #             }, ckpt)
# #             if self.verbose:
# #                 print(f"  ✅ Best val ADE: {ade_km:.1f} km  →  saved {ckpt}")
# #         else:
# #             self.counter += 1
# #             if self.verbose:
# #                 print(f"  EarlyStopping: {self.counter}/{self.patience}  "
# #                       f"(best val ADE={self.best_ade:.1f} km)")
# #             if self.counter >= self.patience:
# #                 self.early_stop = True


# # # ── Main ──────────────────────────────────────────────────────────────────────
# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
# #         device = torch.device('cuda')
# #     else:
# #         device = torch.device('cpu')

# #     os.makedirs(args.output_dir, exist_ok=True)

# #     # ── Header ────────────────────────────────────────────────────────
# #     print("=" * 70)
# #     print("  TC-FlowMatching v4  |  OT-CFM + PINN Vorticity (NS)")
# #     print("=" * 70)
# #     print(f"  Device      : {device}")
# #     print(f"  ODE steps   : {args.ode_steps}")
# #     print(f"  sigma_min   : {args.sigma_min}  (OT path, near-deterministic)")
# #     print(f"  Ensemble    : {args.val_ensemble}")
# #     print(f"  LR          : {args.g_learning_rate}  WD: {args.weight_decay}")
# #     print(f"  Epochs      : {args.num_epochs}  Patience: {args.patience}")
# #     print("=" * 70)

# #     # ── Data ──────────────────────────────────────────────────────────
# #     train_dir, val_dir, test_dir = resolve_data_path(args.dataset_root)

# #     print(f"\n  Data paths:")
# #     print(f"    train : {train_dir}")
# #     print(f"    val   : {val_dir}")
# #     print(f"    test  : {test_dir}  (held-out, year={args.test_year})")

# #     _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)

# #     # ── Val loader: dùng val/ folder, KHÔNG filter theo year ──────────
# #     val_loader = None
# #     if os.path.exists(val_dir):
# #         _, val_loader = data_loader(args, {'root': val_dir, 'type': 'val'}, test=True)
# #         print(f"\n  ✅ Val loader: val/ folder  ({len(val_loader.dataset)} seq)")
# #     else:
# #         # Fallback: nếu không có val/ thì dùng test/ (cảnh báo)
# #         print(f"\n  ⚠️  val/ folder không tồn tại! Fallback về test/ (không lý tưởng)")
# #         if os.path.exists(test_dir):
# #             _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
# #                                         test=True, test_year=args.test_year)
# #             print(f"  ⚠️  Dùng test/ year={args.test_year} làm val  ({len(val_loader.dataset)} seq)")

# #     # ── Test loader: held-out, chỉ eval cuối training ─────────────────
# #     test_loader = None
# #     if os.path.exists(test_dir):
# #         _, test_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
# #                                      test=True, test_year=args.test_year)
# #         print(f"  ✅ Test loader: test/ year={args.test_year}  ({len(test_loader.dataset)} seq)")
# #     else:
# #         print(f"  ⚠️  test/ folder không tồn tại, bỏ qua final test eval")

# #     print(f"\n  Train: {len(train_loader.dataset)} seq  "
# #           f"Val: {len(val_loader.dataset) if val_loader else 0} seq  "
# #           f"Test: {len(test_loader.dataset) if test_loader else 0} seq\n")

# #     # ── Model ─────────────────────────────────────────────────────────
# #     model = TCFlowMatching(
# #         pred_len  = args.pred_len,
# #         obs_len   = args.obs_len,
# #         sigma_min = args.sigma_min,
# #     ).to(device)

# #     n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  Parameters  : {n_p:,}")
# #     print(f"  Architecture: VelocityField (OT-CFM + PINN Vorticity)\n")

# #     assert hasattr(model, "_ns_pinn_loss"), (
# #         "❌ Model thiếu _ns_pinn_loss! Dùng flow_matching_model_v4.py"
# #     )
# #     assert not hasattr(model.net, "ns_physics"), (
# #         "❌ Model vẫn dùng NavierStokesPhysics MLP (v3)! Dùng flow_matching_model_v4.py"
# #     )
# #     print("  ✅ PINN vorticity (NS) confirmed\n")

# #     # ── Optimizer ─────────────────────────────────────────────────────
# #     optimizer    = optim.AdamW(model.parameters(),
# #                                lr=args.g_learning_rate,
# #                                weight_decay=args.weight_decay)
# #     total_steps  = len(train_loader) * args.num_epochs
# #     warmup_steps = len(train_loader) * args.warmup_epochs
# #     scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
# #     saver        = BestModelSaver(patience=args.patience, verbose=True)

# #     # ── Log files ─────────────────────────────────────────────────────
# #     log_path = os.path.join(args.output_dir, 'training_log.csv')
# #     with open(log_path, 'w') as f:
# #         f.write("epoch,train_loss,val_loss,val_ADE_km,val_FDE_km,"
# #                 "val_12h,val_24h,val_48h,val_72h,sigma_min,"
# #                 "epoch_time_s,sample_ms_per_batch,"
# #                 "fm_loss,dir_loss,smooth_loss,disp_loss,curv_loss,pinn_loss,ns_cons\n")

# #     print("=" * 70 + "\n  TRAINING\n" + "=" * 70)

# #     epoch_times = []

# #     for epoch in range(args.num_epochs):

# #         # ── Train ──────────────────────────────────────────────────────
# #         model.train()
# #         train_loss  = 0.0
# #         loss_accum  = {'fm': 0, 'dir': 0, 'smooth': 0,
# #                        'disp': 0, 'curv': 0, 'ns': 0, 'ns_cons': 0}
# #         t_epoch_start = time.time()

# #         for i, batch in enumerate(train_loader):
# #             bl   = move_batch(list(batch), device)
# #             bd   = compute_loss_breakdown(model, bl)
# #             loss = bd['total']

# #             optimizer.zero_grad()
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #             optimizer.step()
# #             scheduler.step()

# #             train_loss += loss.item()
# #             for k in loss_accum:
# #                 loss_accum[k] += bd[k]

# #             if i % 20 == 0:
# #                 lr = optimizer.param_groups[0]['lr']
# #                 print(f"  [{epoch:>3}/{args.num_epochs}] [{i:>3}/{len(train_loader)}]"
# #                       f"  total={loss.item():.4f}"
# #                       f"  fm={bd['fm']:.3f}"
# #                       f"  dir={bd['dir']:.3f}"
# #                       f"  pinn={bd['ns']:.4f}"
# #                       f"  lr={lr:.2e}")

# #         epoch_time    = time.time() - t_epoch_start
# #         epoch_times.append(epoch_time)
# #         avg_train     = train_loss / len(train_loader)
# #         n_bat         = len(train_loader)
# #         avg_breakdown = {k: v / n_bat for k, v in loss_accum.items()}

# #         # ── Validate trên val/ ─────────────────────────────────────────
# #         if val_loader:
# #             model.eval()
# #             val_loss = 0.0
# #             with torch.no_grad():
# #                 for batch in val_loader:
# #                     bl = move_batch(list(batch), device)
# #                     val_loss += model.get_loss(bl).item()
# #             avg_val = val_loss / len(val_loader)

# #             if epoch % args.val_freq == 0 or epoch < 5:
# #                 m, per_step = evaluate_km(
# #                     model, val_loader, device,
# #                     num_ensemble=args.val_ensemble,
# #                     ddim_steps=args.ode_steps,
# #                     pred_len=args.pred_len,
# #                 )
# #                 ade = m['ADE']
# #                 fde = m['FDE']
# #                 sms = m['sample_ms_per_batch']

# #                 print(f"\n{'─'*70}")
# #                 print(f"  Epoch {epoch:>3}  │  train={avg_train:.4f}  val={avg_val:.4f}"
# #                       f"  time={epoch_time:.1f}s")
# #                 print(f"  Loss breakdown │"
# #                       f"  fm={avg_breakdown['fm']:.3f}"
# #                       f"  dir={avg_breakdown['dir']:.3f}"
# #                       f"  disp={avg_breakdown['disp']:.3f}"
# #                       f"  curv={avg_breakdown['curv']:.5f}"
# #                       f"  pinn(NS)={avg_breakdown['ns']:.4f}")
# #                 print(f"  Val (km)       │"
# #                       f"  ADE={ade:.1f}  FDE={fde:.1f}"
# #                       f"  12h={m.get('12h',0):.0f}"
# #                       f"  24h={m.get('24h',0):.0f}"
# #                       f"  48h={m.get('48h',0):.0f}"
# #                       f"  72h={m.get('72h',0):.0f}")
# #                 print(f"  Speed          │"
# #                       f"  sample={sms:.1f}ms/batch"
# #                       f"  epoch={epoch_time:.1f}s"
# #                       f"  avg={sum(epoch_times)/len(epoch_times):.1f}s")
# #                 print(f"{'─'*70}\n")

# #                 for threshold, msg in [
# #                     (500, '📉 ADE<500'), (300, '📉 ADE<300'),
# #                     (200, '🎯 ADE<200'), (150, '🏆 ADE<150'),
# #                     (100, '🌟 ADE<100!'), (50, '🔥 ADE<50km!!!')
# #                 ]:
# #                     if ade < threshold:
# #                         print(f"  {msg} km"); break

# #                 with open(log_path, 'a') as f:
# #                     f.write(
# #                         f"{epoch},{avg_train:.6f},{avg_val:.6f},"
# #                         f"{ade:.1f},{fde:.1f},"
# #                         f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
# #                         f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
# #                         f"{args.sigma_min:.4f},"
# #                         f"{epoch_time:.1f},{sms:.1f},"
# #                         f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
# #                         f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
# #                         f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
# #                         f"{avg_breakdown['ns_cons']:.4f}\n"
# #                     )

# #                 # Early stopping + best model dựa trên val ADE
# #                 saver(ade, model, args.output_dir, epoch,
# #                       optimizer, avg_train, avg_val)

# #             else:
# #                 print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
# #                       f"  val={avg_val:.4f}  time={epoch_time:.1f}s\n")
# #                 with open(log_path, 'a') as f:
# #                     f.write(
# #                         f"{epoch},{avg_train:.6f},{avg_val:.6f},"
# #                         f",,,,,,"
# #                         f"{args.sigma_min:.4f},"
# #                         f"{epoch_time:.1f},,"
# #                         f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
# #                         f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
# #                         f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
# #                         f"{avg_breakdown['ns_cons']:.4f}\n"
# #                     )

# #             if saver.early_stop:
# #                 print("  Early stopping triggered."); break

# #         else:
# #             print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
# #                   f"  time={epoch_time:.1f}s\n")

# #         if (epoch + 1) % args.save_interval == 0:
# #             ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
# #             torch.save({'epoch': epoch,
# #                         'model_state_dict': model.state_dict()}, ckpt)
# #             print(f"  ✓ Checkpoint → {ckpt}\n")

# #     # ── Final eval trên test/ (held-out) ──────────────────────────────
# #     print(f"\n{'='*70}")
# #     print(f"  FINAL TEST EVALUATION  (held-out, year={args.test_year})")
# #     print(f"{'='*70}")

# #     if test_loader:
# #         # Load best model
# #         best_ckpt = os.path.join(args.output_dir, 'best_model.pth')
# #         if os.path.exists(best_ckpt):
# #             ckpt_data = torch.load(best_ckpt, map_location=device)
# #             model.load_state_dict(ckpt_data['model_state_dict'])
# #             print(f"  Loaded best model from epoch {ckpt_data['epoch']}"
# #                   f"  (val ADE={ckpt_data['val_ade_km']:.1f} km)")

# #         test_m, _ = evaluate_km(
# #             model, test_loader, device,
# #             num_ensemble=args.val_ensemble,
# #             ddim_steps=args.ode_steps,
# #             pred_len=args.pred_len,
# #         )
# #         print(f"\n  Test (km) │"
# #               f"  ADE={test_m['ADE']:.1f}  FDE={test_m['FDE']:.1f}"
# #               f"  12h={test_m.get('12h',0):.0f}"
# #               f"  24h={test_m.get('24h',0):.0f}"
# #               f"  48h={test_m.get('48h',0):.0f}"
# #               f"  72h={test_m.get('72h',0):.0f}")

# #         # Ghi test results ra file riêng
# #         test_log = os.path.join(args.output_dir, 'test_results.txt')
# #         with open(test_log, 'w') as f:
# #             f.write(f"Test year : {args.test_year}\n")
# #             f.write(f"ADE (km)  : {test_m['ADE']:.1f}\n")
# #             f.write(f"FDE (km)  : {test_m['FDE']:.1f}\n")
# #             f.write(f"12h (km)  : {test_m.get('12h',0):.1f}\n")
# #             f.write(f"24h (km)  : {test_m.get('24h',0):.1f}\n")
# #             f.write(f"48h (km)  : {test_m.get('48h',0):.1f}\n")
# #             f.write(f"72h (km)  : {test_m.get('72h',0):.1f}\n")
# #         print(f"\n  Test results → {test_log}")
# #     else:
# #         print("  ⚠️  Không có test loader, bỏ qua.")

# #     avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
# #     print(f"\n{'='*70}")
# #     print(f"  DONE")
# #     print(f"  Best val ADE    : {saver.best_ade:.1f} km")
# #     print(f"  Avg epoch time  : {avg_epoch_time:.1f}s")
# #     print(f"  Train log       : {log_path}")
# #     print(f"{'='*70}\n")


# # if __name__ == '__main__':
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed(42)
# #     main(args)

# """
# scripts/train_flowmatching.py
# ==============================
# Training TCFlowMatching v4 với OT-CFM + PINN Vorticity.

# Metrics:
#   - ADE/FDE/24h/48h/72h (km)
#   - Train time per epoch (s)
#   - Sample time per batch (ms)
#   - Loss breakdown: fm, dir, smooth, disp, curv, pinn(NS)

# Data split:
#   - train/  : training data
#   - val/    : validation (early stopping, best model selection)
#   - test/   : held-out test set (chỉ đánh giá cuối cùng, KHÔNG dùng để tune)

# Chạy:
#   python scripts/train_flowmatching.py \
#       --dataset_root TCND_vn \
#       --output_dir   model_save/flowmatching_v4 \
#       --ode_steps 10 --sigma_min 0.001 \
#       --num_epochs 200 --batch_size 32
# """

# import argparse, os, sys, time, math
# import torch, torch.optim as optim
# import torch.nn.functional as F
# import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from TCNM.data.loader import data_loader
# from TCNM.flow_matching_model import TCFlowMatching
# from TCNM.utils import get_cosine_schedule_with_warmup


# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--dataset_root',        default='TCND_vn',                  type=str)
#     p.add_argument('--obs_len',             default=8,                           type=int)
#     p.add_argument('--pred_len',            default=12,                          type=int)
#     p.add_argument('--batch_size',          default=32,                          type=int)
#     p.add_argument('--num_epochs',          default=200,                         type=int)
#     p.add_argument('--g_learning_rate',     default=2e-4,                        type=float)
#     p.add_argument('--weight_decay',        default=1e-4,                        type=float)
#     p.add_argument('--warmup_epochs',       default=5,                           type=int)
#     p.add_argument('--grad_clip',           default=1.0,                         type=float)
#     p.add_argument('--patience',            default=40,                          type=int)
#     p.add_argument('--gpu_num',             default='0',                         type=str)
#     p.add_argument('--output_dir',          default='model_save/flowmatching_v4',type=str)
#     p.add_argument('--save_interval',       default=10,                          type=int)
#     p.add_argument('--test_year',           default=2019,                        type=int,
#                    help='Năm để filter test set (held-out, chỉ eval cuối)')
#     p.add_argument('--val_ensemble',        default=5,                           type=int)
#     p.add_argument('--ode_steps',           default=10,                          type=int,
#                    help='ODE integration steps cho Flow Matching')
#     p.add_argument('--val_freq',            default=5,                           type=int)
#     p.add_argument('--sigma_min',           default=0.001,                       type=float)
#     # compat
#     p.add_argument('--d_model',    default=128,   type=int)
#     p.add_argument('--delim',      default=' ')
#     p.add_argument('--skip',       default=1,     type=int)
#     p.add_argument('--min_ped',    default=1,     type=int)
#     p.add_argument('--threshold',  default=0.002, type=float)
#     p.add_argument('--other_modal',default='gph')
#     return p.parse_args()


# def resolve_data_path(root):
#     """
#     Trả về (train_dir, val_dir, test_dir).

#     Ưu tiên dùng val/ folder riêng biệt.
#     Nếu không có val/, fallback về test/ (với cảnh báo).
#     """
#     root = root.rstrip('/\\')

#     # Nếu truyền vào thẳng Data1d/train hoặc tương tự
#     if root.endswith(('Data1d/train', 'Data1d\\train')):
#         base = root[:-len('train')]
#     elif root.endswith(('Data1d/test', 'Data1d\\test')):
#         base = root[:-len('test')]
#     elif root.endswith('Data1d'):
#         base = root + os.sep
#     else:
#         base = os.path.join(root, 'Data1d') + os.sep

#     train_dir = os.path.join(base, 'train')
#     val_dir   = os.path.join(base, 'val')
#     test_dir  = os.path.join(base, 'test')

#     return train_dir, val_dir, test_dir


# def move_batch(bl, device):
#     for j, x in enumerate(bl):
#         if torch.is_tensor(x):
#             bl[j] = x.to(device)
#         elif isinstance(x, dict):
#             bl[j] = {k: v.to(device) if torch.is_tensor(v) else v
#                      for k, v in x.items()}
#     return bl


# def denorm_traj(n):
#     r = n.clone()
#     r[..., 0] = n[..., 0] * 50 + 1800
#     r[..., 1] = n[..., 1] * 50
#     return r


# # ── Loss breakdown ────────────────────────────────────────────────────────────
# def compute_loss_breakdown(model, batch_list):
#     traj_gt = batch_list[1]
#     Me_gt   = batch_list[8]
#     obs     = batch_list[0]
#     obs_Me  = batch_list[7]

#     B      = traj_gt.shape[1]
#     device = traj_gt.device
#     lp, lm = obs[-1], obs_Me[-1]
#     sm     = model.sigma_min

#     x1    = model.traj_to_rel(traj_gt, Me_gt, lp, lm)
#     x0    = torch.randn_like(x1) * sm
#     t     = torch.rand(B, device=device)
#     t_exp = t.view(B, 1, 1)

#     x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
#     denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
#     target_vel = (x1 - (1 - sm) * x_t) / denom

#     pred_vel    = model.net(x_t, t, batch_list)
#     fm_loss     = F.mse_loss(pred_vel, target_vel)

#     pred_x1     = x_t + denom * pred_vel
#     pred_abs, _ = model.rel_to_abs(pred_x1, lp, lm)

#     overall_dir = model._overall_dir_loss(pred_abs, traj_gt, lp)
#     step_dir    = model._step_dir_loss(pred_abs, traj_gt)
#     disp_l      = model._weighted_disp_loss(pred_abs, traj_gt)
#     heading_l   = model._heading_change_loss(pred_abs, traj_gt)
#     smt_l       = model._smooth_loss(pred_abs)
#     pinn_l      = model._ns_pinn_loss(pred_abs)

#     total = (  1.0 * fm_loss
#              + 1.5 * overall_dir
#              + 1.5 * step_dir
#              + 1.0 * disp_l
#              + 2.0 * heading_l
#              + 0.2 * smt_l
#              + 0.5 * pinn_l)

#     return {
#         'total':   total,
#         'fm':      fm_loss.item(),
#         'dir':     (overall_dir + step_dir).item(),
#         'smooth':  smt_l.item(),
#         'disp':    disp_l.item(),
#         'curv':    heading_l.item(),
#         'ns':      pinn_l.item(),
#         'ns_cons': 0.0,
#     }


# # ── Validation / Test metrics ─────────────────────────────────────────────────
# def evaluate_km(model, loader, device, num_ensemble=5,
#                 ddim_steps=10, pred_len=12):
#     model.eval()
#     all_step_errors = []
#     total_sample_ms = 0.0
#     n_batches       = 0

#     with torch.no_grad():
#         for batch in loader:
#             bl = move_batch(list(batch), device)
#             gt = bl[1]

#             t0 = time.time()
#             pred, _ = model.sample(bl, num_ensemble=num_ensemble,
#                                    ddim_steps=ddim_steps)
#             total_sample_ms += (time.time() - t0) * 1000
#             n_batches += 1

#             pred_r = denorm_traj(pred)
#             gt_r   = denorm_traj(gt)
#             dist   = torch.norm(pred_r - gt_r, dim=2) * 11.1
#             all_step_errors.append(dist.mean(dim=1).cpu())

#     stacked    = torch.stack(all_step_errors, dim=0)
#     mean_steps = stacked.mean(dim=0)

#     m = {'ADE': mean_steps.mean().item(), 'FDE': mean_steps[-1].item()}
#     for h, s in [(12, 1), (24, 3), (48, 7), (72, 11)]:
#         if s < pred_len:
#             m[f'{h}h'] = mean_steps[s].item()

#     m['sample_ms_per_batch'] = total_sample_ms / max(n_batches, 1)
#     return m, mean_steps


# # ── Best Model Saver ──────────────────────────────────────────────────────────
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
#                 'model_type':       'TCFlowMatching_v4_OT-CFM+PINN',
#                 'args': {
#                     'obs_len':   model.obs_len,
#                     'pred_len':  model.pred_len,
#                     'sigma_min': model.sigma_min,
#                 },
#             }, ckpt)
#             if self.verbose:
#                 print(f"  ✅ Best val ADE: {ade_km:.1f} km  →  saved {ckpt}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"  EarlyStopping: {self.counter}/{self.patience}  "
#                       f"(best val ADE={self.best_ade:.1f} km)")
#             if self.counter >= self.patience:
#                 self.early_stop = True


# # ── Main ──────────────────────────────────────────────────────────────────────
# def main(args):
#     if torch.cuda.is_available():
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     os.makedirs(args.output_dir, exist_ok=True)

#     # ── Header ────────────────────────────────────────────────────────
#     print("=" * 70)
#     print("  TC-FlowMatching v4  |  OT-CFM + PINN Vorticity (NS)")
#     print("=" * 70)
#     print(f"  Device      : {device}")
#     print(f"  ODE steps   : {args.ode_steps}")
#     print(f"  sigma_min   : {args.sigma_min}  (OT path, near-deterministic)")
#     print(f"  Ensemble    : {args.val_ensemble}")
#     print(f"  LR          : {args.g_learning_rate}  WD: {args.weight_decay}")
#     print(f"  Epochs      : {args.num_epochs}  Patience: {args.patience}")
#     print("=" * 70)

#     # ── Data ──────────────────────────────────────────────────────────
#     train_dir, val_dir, test_dir = resolve_data_path(args.dataset_root)

#     print(f"\n  Data paths:")
#     print(f"    train : {train_dir}")
#     print(f"    val   : {val_dir}")
#     print(f"    test  : {test_dir}  (held-out, year={args.test_year})")

#     _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)

#     # ── Val loader: dùng val/ folder, KHÔNG filter theo year ──────────
#     val_loader = None
#     if os.path.exists(val_dir):
#         _, val_loader = data_loader(args, {'root': val_dir, 'type': 'val'}, test=True)
#         print(f"\n  ✅ Val loader: val/ folder  ({len(val_loader.dataset)} seq)")
#     else:
#         # Fallback: nếu không có val/ thì dùng test/ (cảnh báo)
#         print(f"\n  ⚠️  val/ folder không tồn tại! Fallback về test/ (không lý tưởng)")
#         if os.path.exists(test_dir):
#             _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
#                                         test=True, test_year=args.test_year)
#             print(f"  ⚠️  Dùng test/ year={args.test_year} làm val  ({len(val_loader.dataset)} seq)")

#     # ── Test loader: held-out, chỉ eval cuối training ─────────────────
#     test_loader = None
#     if os.path.exists(test_dir):
#         _, test_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
#                                      test=True, test_year=args.test_year)
#         print(f"  ✅ Test loader: test/ year={args.test_year}  ({len(test_loader.dataset)} seq)")
#     else:
#         print(f"  ⚠️  test/ folder không tồn tại, bỏ qua final test eval")

#     print(f"\n  Train: {len(train_loader.dataset)} seq  "
#           f"Val: {len(val_loader.dataset) if val_loader else 0} seq  "
#           f"Test: {len(test_loader.dataset) if test_loader else 0} seq\n")

#     # ── Model ─────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len  = args.pred_len,
#         obs_len   = args.obs_len,
#         sigma_min = args.sigma_min,
#     ).to(device)

#     n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  Parameters  : {n_p:,}")
#     print(f"  Architecture: VelocityField (OT-CFM + PINN Vorticity)\n")

#     assert hasattr(model, "_ns_pinn_loss"), (
#         "❌ Model thiếu _ns_pinn_loss! Dùng flow_matching_model_v4.py"
#     )
#     assert not hasattr(model.net, "ns_physics"), (
#         "❌ Model vẫn dùng NavierStokesPhysics MLP (v3)! Dùng flow_matching_model_v4.py"
#     )
#     print("  ✅ PINN vorticity (NS) confirmed\n")

#     # ── Optimizer ─────────────────────────────────────────────────────
#     optimizer    = optim.AdamW(model.parameters(),
#                                lr=args.g_learning_rate,
#                                weight_decay=args.weight_decay)
#     total_steps  = len(train_loader) * args.num_epochs
#     warmup_steps = len(train_loader) * args.warmup_epochs
#     scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
#     saver        = BestModelSaver(patience=args.patience, verbose=True)

#     # ── Log files ─────────────────────────────────────────────────────
#     log_path = os.path.join(args.output_dir, 'training_log.csv')
#     with open(log_path, 'w') as f:
#         f.write("epoch,train_loss,val_loss,val_ADE_km,val_FDE_km,"
#                 "val_12h,val_24h,val_48h,val_72h,sigma_min,"
#                 "epoch_time_s,sample_ms_per_batch,"
#                 "fm_loss,dir_loss,smooth_loss,disp_loss,curv_loss,pinn_loss,ns_cons\n")

#     print("=" * 70 + "\n  TRAINING\n" + "=" * 70)

#     epoch_times = []

#     for epoch in range(args.num_epochs):

#         # ── Train ──────────────────────────────────────────────────────
#         model.train()
#         train_loss  = 0.0
#         loss_accum  = {'fm': 0, 'dir': 0, 'smooth': 0,
#                        'disp': 0, 'curv': 0, 'ns': 0, 'ns_cons': 0}
#         t_epoch_start = time.time()

#         for i, batch in enumerate(train_loader):
#             bl   = move_batch(list(batch), device)
#             bd   = compute_loss_breakdown(model, bl)
#             loss = bd['total']

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()
#             scheduler.step()

#             train_loss += loss.item()
#             for k in loss_accum:
#                 loss_accum[k] += bd[k]

#             if i % 20 == 0:
#                 lr = optimizer.param_groups[0]['lr']
#                 print(f"  [{epoch:>3}/{args.num_epochs}] [{i:>3}/{len(train_loader)}]"
#                       f"  total={loss.item():.4f}"
#                       f"  fm={bd['fm']:.3f}"
#                       f"  dir={bd['dir']:.3f}"
#                       f"  pinn={bd['ns']:.4f}"
#                       f"  lr={lr:.2e}")

#         epoch_time    = time.time() - t_epoch_start
#         epoch_times.append(epoch_time)
#         avg_train     = train_loss / len(train_loader)
#         n_bat         = len(train_loader)
#         avg_breakdown = {k: v / n_bat for k, v in loss_accum.items()}

#         # ── Validate trên val/ ─────────────────────────────────────────
#         if val_loader:
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     bl = move_batch(list(batch), device)
#                     val_loss += model.get_loss(bl).item()
#             avg_val = val_loss / len(val_loader)

#             if epoch % args.val_freq == 0 or epoch < 5:
#                 m, per_step = evaluate_km(
#                     model, val_loader, device,
#                     num_ensemble=args.val_ensemble,
#                     ddim_steps=args.ode_steps,
#                     pred_len=args.pred_len,
#                 )
#                 ade = m['ADE']
#                 fde = m['FDE']
#                 sms = m['sample_ms_per_batch']

#                 print(f"\n{'─'*70}")
#                 print(f"  Epoch {epoch:>3}  │  train={avg_train:.4f}  val={avg_val:.4f}"
#                       f"  time={epoch_time:.1f}s")
#                 print(f"  Loss breakdown │"
#                       f"  fm={avg_breakdown['fm']:.3f}"
#                       f"  dir={avg_breakdown['dir']:.3f}"
#                       f"  disp={avg_breakdown['disp']:.3f}"
#                       f"  curv={avg_breakdown['curv']:.3f}"
#                       f"  pinn(NS)={avg_breakdown['ns']:.4f}")
#                 print(f"  Val (km)       │"
#                       f"  ADE={ade:.1f}  FDE={fde:.1f}"
#                       f"  12h={m.get('12h',0):.0f}"
#                       f"  24h={m.get('24h',0):.0f}"
#                       f"  48h={m.get('48h',0):.0f}"
#                       f"  72h={m.get('72h',0):.0f}")
#                 print(f"  Speed          │"
#                       f"  sample={sms:.1f}ms/batch"
#                       f"  epoch={epoch_time:.1f}s"
#                       f"  avg={sum(epoch_times)/len(epoch_times):.1f}s")
#                 print(f"{'─'*70}\n")

#                 for threshold, msg in [
#                     (500, '📉 ADE<500'), (300, '📉 ADE<300'),
#                     (200, '🎯 ADE<200'), (150, '🏆 ADE<150'),
#                     (100, '🌟 ADE<100!'), (50, '🔥 ADE<50km!!!')
#                 ]:
#                     if ade < threshold:
#                         print(f"  {msg} km"); break

#                 with open(log_path, 'a') as f:
#                     f.write(
#                         f"{epoch},{avg_train:.6f},{avg_val:.6f},"
#                         f"{ade:.1f},{fde:.1f},"
#                         f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
#                         f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
#                         f"{args.sigma_min:.4f},"
#                         f"{epoch_time:.1f},{sms:.1f},"
#                         f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
#                         f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
#                         f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
#                         f"{avg_breakdown['ns_cons']:.4f}\n"
#                     )

#                 # Early stopping + best model dựa trên val ADE
#                 saver(ade, model, args.output_dir, epoch,
#                       optimizer, avg_train, avg_val)

#             else:
#                 print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
#                       f"  val={avg_val:.4f}  time={epoch_time:.1f}s\n")
#                 with open(log_path, 'a') as f:
#                     f.write(
#                         f"{epoch},{avg_train:.6f},{avg_val:.6f},"
#                         f",,,,,,"
#                         f"{args.sigma_min:.4f},"
#                         f"{epoch_time:.1f},,"
#                         f"{avg_breakdown['fm']:.4f},{avg_breakdown['dir']:.4f},"
#                         f"{avg_breakdown['smooth']:.4f},{avg_breakdown['disp']:.4f},"
#                         f"{avg_breakdown['curv']:.4f},{avg_breakdown['ns']:.4f},"
#                         f"{avg_breakdown['ns_cons']:.4f}\n"
#                     )

#             if saver.early_stop:
#                 print("  Early stopping triggered."); break

#         else:
#             print(f"\n  Epoch {epoch:>3}  │  train={avg_train:.4f}"
#                   f"  time={epoch_time:.1f}s\n")

#         if (epoch + 1) % args.save_interval == 0:
#             ckpt = os.path.join(args.output_dir, f'ckpt_{epoch}.pth')
#             torch.save({'epoch': epoch,
#                         'model_state_dict': model.state_dict()}, ckpt)
#             print(f"  ✓ Checkpoint → {ckpt}\n")

#     # ── Final eval trên test/ (held-out) ──────────────────────────────
#     print(f"\n{'='*70}")
#     print(f"  FINAL TEST EVALUATION  (held-out, year={args.test_year})")
#     print(f"{'='*70}")

#     if test_loader:
#         # Load best model
#         best_ckpt = os.path.join(args.output_dir, 'best_model.pth')
#         if os.path.exists(best_ckpt):
#             ckpt_data = torch.load(best_ckpt, map_location=device)
#             model.load_state_dict(ckpt_data['model_state_dict'])
#             print(f"  Loaded best model from epoch {ckpt_data['epoch']}"
#                   f"  (val ADE={ckpt_data['val_ade_km']:.1f} km)")

#         test_m, _ = evaluate_km(
#             model, test_loader, device,
#             num_ensemble=args.val_ensemble,
#             ddim_steps=args.ode_steps,
#             pred_len=args.pred_len,
#         )
#         print(f"\n  Test (km) │"
#               f"  ADE={test_m['ADE']:.1f}  FDE={test_m['FDE']:.1f}"
#               f"  12h={test_m.get('12h',0):.0f}"
#               f"  24h={test_m.get('24h',0):.0f}"
#               f"  48h={test_m.get('48h',0):.0f}"
#               f"  72h={test_m.get('72h',0):.0f}")

#         # Ghi test results ra file riêng
#         test_log = os.path.join(args.output_dir, 'test_results.txt')
#         with open(test_log, 'w') as f:
#             f.write(f"Test year : {args.test_year}\n")
#             f.write(f"ADE (km)  : {test_m['ADE']:.1f}\n")
#             f.write(f"FDE (km)  : {test_m['FDE']:.1f}\n")
#             f.write(f"12h (km)  : {test_m.get('12h',0):.1f}\n")
#             f.write(f"24h (km)  : {test_m.get('24h',0):.1f}\n")
#             f.write(f"48h (km)  : {test_m.get('48h',0):.1f}\n")
#             f.write(f"72h (km)  : {test_m.get('72h',0):.1f}\n")
#         print(f"\n  Test results → {test_log}")
#     else:
#         test_m = None
#         print("  ⚠️  Không có test loader, bỏ qua.")

#     avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

#     # ── Summary CSV ───────────────────────────────────────────────────
#     # Đọc lại training_log.csv để tổng hợp
#     import csv, pandas as pd

#     summary_path = os.path.join(args.output_dir, 'summary.csv')
#     try:
#         df = pd.read_csv(log_path)

#         # Lấy epoch tốt nhất theo val_ADE_km
#         df_eval = df.dropna(subset=['val_ADE_km'])
#         best_epoch_row = df_eval.loc[df_eval['val_ADE_km'].idxmin()]
#         best_epoch = int(best_epoch_row['epoch'])

#         # Thêm cột split để phân biệt train log vs test result
#         df['split'] = 'val'

#         # Thêm dòng test result (nếu có)
#         if test_m is not None:
#             best_ckpt_data = torch.load(
#                 os.path.join(args.output_dir, 'best_model.pth'),
#                 map_location='cpu'
#             )
#             test_row = {
#                 'epoch':              best_ckpt_data.get('epoch', best_epoch),
#                 'train_loss':         best_epoch_row.get('train_loss', ''),
#                 'val_loss':           best_epoch_row.get('val_loss', ''),
#                 'val_ADE_km':         '',
#                 'val_FDE_km':         '',
#                 'val_12h':            '',
#                 'val_24h':            '',
#                 'val_48h':            '',
#                 'val_72h':            '',
#                 'sigma_min':          args.sigma_min,
#                 'epoch_time_s':       '',
#                 'sample_ms_per_batch':'',
#                 'fm_loss':            best_epoch_row.get('fm_loss', ''),
#                 'dir_loss':           best_epoch_row.get('dir_loss', ''),
#                 'smooth_loss':        best_epoch_row.get('smooth_loss', ''),
#                 'disp_loss':          best_epoch_row.get('disp_loss', ''),
#                 'curv_loss':          best_epoch_row.get('curv_loss', ''),
#                 'pinn_loss':          best_epoch_row.get('pinn_loss', ''),
#                 'ns_cons':            best_epoch_row.get('ns_cons', ''),
#                 # Test-only columns
#                 'test_ADE_km':        test_m['ADE'],
#                 'test_FDE_km':        test_m['FDE'],
#                 'test_12h':           test_m.get('12h', ''),
#                 'test_24h':           test_m.get('24h', ''),
#                 'test_48h':           test_m.get('48h', ''),
#                 'test_72h':           test_m.get('72h', ''),
#                 'split':              'test',
#             }
#             df = pd.concat([df, pd.DataFrame([test_row])], ignore_index=True)

#         # Thêm metadata header bằng cách ghi tay
#         with open(summary_path, 'w', newline='') as f:
#             # Block 1: Run config
#             f.write("# ── Run Config ──\n")
#             f.write(f"# model,TC-FlowMatching v4 OT-CFM+PINN\n")
#             f.write(f"# dataset_root,{args.dataset_root}\n")
#             f.write(f"# obs_len,{args.obs_len}\n")
#             f.write(f"# pred_len,{args.pred_len}\n")
#             f.write(f"# batch_size,{args.batch_size}\n")
#             f.write(f"# num_epochs,{args.num_epochs}\n")
#             f.write(f"# lr,{args.g_learning_rate}\n")
#             f.write(f"# weight_decay,{args.weight_decay}\n")
#             f.write(f"# sigma_min,{args.sigma_min}\n")
#             f.write(f"# ode_steps,{args.ode_steps}\n")
#             f.write(f"# val_ensemble,{args.val_ensemble}\n")
#             f.write(f"# patience,{args.patience}\n")
#             f.write(f"# test_year,{args.test_year}\n")
#             f.write("#\n")

#             # Block 2: Best results
#             f.write("# ── Best Results ──\n")
#             f.write(f"# best_epoch,{best_epoch}\n")
#             f.write(f"# best_val_ADE_km,{saver.best_ade:.1f}\n")
#             if test_m:
#                 f.write(f"# test_ADE_km,{test_m['ADE']:.1f}\n")
#                 f.write(f"# test_FDE_km,{test_m['FDE']:.1f}\n")
#                 f.write(f"# test_12h,{test_m.get('12h',0):.1f}\n")
#                 f.write(f"# test_24h,{test_m.get('24h',0):.1f}\n")
#                 f.write(f"# test_48h,{test_m.get('48h',0):.1f}\n")
#                 f.write(f"# test_72h,{test_m.get('72h',0):.1f}\n")
#             f.write(f"# avg_epoch_time_s,{avg_epoch_time:.1f}\n")
#             f.write(f"# total_epochs_trained,{len(epoch_times)}\n")
#             f.write("#\n")

#         # Append epoch-by-epoch data
#         df.to_csv(summary_path, mode='a', index=False)
#         print(f"\n  Summary CSV → {summary_path}")

#     except Exception as e:
#         print(f"\n  ⚠️  Không thể tạo summary CSV: {e}")

#     print(f"\n{'='*70}")
#     print(f"  DONE")
#     print(f"  Best val ADE    : {saver.best_ade:.1f} km")
#     if test_m:
#         print(f"  Test ADE        : {test_m['ADE']:.1f} km  (year={args.test_year})")
#     print(f"  Avg epoch time  : {avg_epoch_time:.1f}s")
#     print(f"  Train log       : {log_path}")
#     print(f"  Summary CSV     : {summary_path}")
#     print(f"{'='*70}\n")


# if __name__ == '__main__':
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(42)
#     main(args)

"""
scripts/train_flowmatching.py
==============================
Training script for TCFlowMatching v7.

Loss weights per Eq. (60):
    FM=1.0  overall_dir=2.0  step_dir=0.5  disp=1.0
    heading=2.0  smooth=0.2  pinn=0.5

Run:
    python scripts/train_flowmatching.py \\
        --dataset_root TCND_vn \\
        --output_dir   model_save/flowmatching_v7 \\
        --sigma_min 0.02 --ode_steps 10 \\
        --num_epochs 200 --batch_size 32
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from TCNM.data.loader import data_loader
from TCNM.flow_matching_model import TCFlowMatching
from TCNM.utils import get_cosine_schedule_with_warmup


# ── CLI arguments ──────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument('--dataset_root',    default='TCND_vn',                   type=str)
    p.add_argument('--obs_len',         default=8,                            type=int)
    p.add_argument('--pred_len',        default=12,                           type=int)
    p.add_argument('--test_year',       default=2019,                         type=int)

    # Training
    p.add_argument('--batch_size',      default=32,                           type=int)
    p.add_argument('--num_epochs',      default=200,                          type=int)
    p.add_argument('--g_learning_rate', default=2e-4,                         type=float)
    p.add_argument('--weight_decay',    default=1e-4,                         type=float)
    p.add_argument('--warmup_epochs',   default=5,                            type=int)
    p.add_argument('--grad_clip',       default=1.0,                          type=float)
    p.add_argument('--patience',        default=40,                           type=int)

    # Model
    p.add_argument('--sigma_min',       default=0.02,                         type=float)
    p.add_argument('--ode_steps',       default=10,                           type=int)
    p.add_argument('--val_ensemble',    default=5,                            type=int)

    # Logging
    p.add_argument('--output_dir',      default='model_save/flowmatching_v7', type=str)
    p.add_argument('--save_interval',   default=10,                           type=int)
    p.add_argument('--val_freq',        default=5,                            type=int)
    p.add_argument('--gpu_num',         default='0',                          type=str)

    # Dataset loader compat flags
    p.add_argument('--d_model',     default=128,   type=int)
    p.add_argument('--delim',       default=' ')
    p.add_argument('--skip',        default=1,     type=int)
    p.add_argument('--min_ped',     default=1,     type=int)
    p.add_argument('--threshold',   default=0.002, type=float)
    p.add_argument('--other_modal', default='gph')

    return p.parse_args()


# ── Data helpers ───────────────────────────────────────────────────────────────

def resolve_data_dirs(root: str):
    root = root.rstrip('/\\')
    if root.endswith(('Data1d/train', 'Data1d\\train')):
        base = root[:-len('train')]
    elif root.endswith('Data1d'):
        base = root + os.sep
    else:
        base = os.path.join(root, 'Data1d') + os.sep
    return (
        os.path.join(base, 'train'),
        os.path.join(base, 'val'),
        os.path.join(base, 'test'),
    )


def move_to(batch, device: torch.device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def denorm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """
    Chuyển normalized coords → degrees.
    Normalization (từ TrajectoryDataset):
        lon_norm = (lon_deg - 180) / 5  →  lon_deg = lon_norm*5 + 180
        lat_norm = lat_deg / 5          →  lat_deg = lat_norm*5
    Dải hợp lệ: lon ∈ [100°E, 180°E], lat ∈ [0°N, 40°N] cho WNP/Biển Đông
    """
    out = t.clone()
    out[..., 0] = t[..., 0] * 5.0 + 180.0  # lon
    out[..., 1] = t[..., 1] * 5.0           # lat
    return out


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model:        TCFlowMatching,
    loader,
    device:       torch.device,
    num_ensemble: int,
    ddim_steps:   int,
    pred_len:     int,
) -> dict:
    """
    Returns ADE, FDE, per-lead-time errors, and inference speed.
    Distance: normalised units × 11.1 km (0.1° ≈ 11.1 km at equator).
    """
    model.eval()
    step_errors = []
    total_ms    = 0.0
    n_batches   = 0

    with torch.no_grad():
        for batch in loader:
            bl = move_to(list(batch), device)
            gt = bl[1]                            # [T_pred, B, 2]

            t0 = time.perf_counter()
            pred, _ = model.sample(bl, num_ensemble=num_ensemble,
                                   ddim_steps=ddim_steps)
            total_ms += (time.perf_counter() - t0) * 1e3
            n_batches += 1

            # pred_deg = denorm_to_deg(pred)
            # gt_deg   = denorm_to_deg(gt)
            # dist_km  = pred_deg.sub(gt_deg).norm(dim=-1).mul(11.1)  # [T, B]
            # step_errors.append(dist_km.mean(dim=1).cpu())            # [T]

            pred_deg = denorm_to_deg(pred)
            gt_deg   = denorm_to_deg(gt)

            # Haversine distance (degrees → km)
            lat1 = torch.deg2rad(pred_deg[..., 1])
            lat2 = torch.deg2rad(gt_deg[..., 1])
            dlat = torch.deg2rad(gt_deg[..., 1] - pred_deg[..., 1])
            dlon = torch.deg2rad(gt_deg[..., 0] - pred_deg[..., 0])
            a    = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
            dist_km = 2 * 6371.0 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))  # [T, B]

            step_errors.append(dist_km.mean(dim=1).cpu())

    mean_steps = torch.stack(step_errors).mean(dim=0)  # [T_pred]

    m = {
        'ADE': mean_steps.mean().item(),
        'FDE': mean_steps[-1].item(),
        'ms_per_batch': total_ms / max(n_batches, 1),
    }
    for h, s in ((12, 1), (24, 3), (48, 7), (72, 11)):
        if s < pred_len:
            m[f'{h}h'] = mean_steps[s].item()

    return m


# ── Checkpoint helper ──────────────────────────────────────────────────────────

class BestModelSaver:
    def __init__(self, patience: int = 40, min_delta: float = 2.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_ade   = float('inf')
        self.counter    = 0
        self.early_stop = False

    def __call__(
        self,
        ade:        float,
        model:      TCFlowMatching,
        out_dir:    str,
        epoch:      int,
        optimizer:  torch.optim.Optimizer,
        train_loss: float,
        val_loss:   float,
    ):
        if ade < self.best_ade - self.min_delta:
            self.best_ade = ade
            self.counter  = 0
            path = os.path.join(out_dir, 'best_model.pth')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state':  optimizer.state_dict(),
                'train_loss':       train_loss,
                'val_loss':         val_loss,
                'val_ade_km':       ade,
                'model_version':    'v7',
                'loss_eq':          'Eq60: FM=1.0 dir=2.0 step=0.5 disp=1.0 heading=2.0 smooth=0.2 pinn=0.5',
                'sigma_min':        model.sigma_min,
                'pred_len':         model.pred_len,
                'obs_len':          model.obs_len,
            }, path)
            print(f"  ✅  Best ADE {ade:.1f} km  →  {path}")
        else:
            self.counter += 1
            print(f"  ⏳  No improvement {self.counter}/{self.patience}"
                  f"  (best={self.best_ade:.1f} km)")
            if self.counter >= self.patience:
                self.early_stop = True


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sigma_min < 0.01:
        print(f"  ⚠️  sigma_min={args.sigma_min} — recommend ≥ 0.02")

    print("=" * 68)
    print("  TC-FlowMatching v7  |  Loss: Eq. (60)")
    print("=" * 68)
    print(f"  device      {device}")
    print(f"  sigma_min   {args.sigma_min}")
    print(f"  ode_steps   {args.ode_steps}   ensemble {args.val_ensemble}")
    print(f"  epochs      {args.num_epochs}   patience {args.patience}")
    print(f"  LR          {args.g_learning_rate}   WD {args.weight_decay}")
    print(f"  Eq.(60)     FM=1.0  dir=2.0  step=0.5  disp=1.0")
    print(f"              heading=2.0  smooth=0.2  pinn=0.5")
    print("=" * 68)

    # ── Data loaders ───────────────────────────────────────────────────────────
    train_dir, val_dir, test_dir = resolve_data_dirs(args.dataset_root)

    _, train_loader = data_loader(args, {'root': train_dir, 'type': 'train'}, test=False)

    val_loader = None
    if os.path.isdir(val_dir):
        _, val_loader = data_loader(args, {'root': val_dir, 'type': 'val'}, test=True)
        print(f"\n  train  {len(train_loader.dataset):>5} seq")
        print(f"  val    {len(val_loader.dataset):>5} seq")
    else:
        print(f"\n  ⚠️  val/ not found — using test/ as validation")
        _, val_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                    test=True, test_year=args.test_year)
        print(f"  train  {len(train_loader.dataset):>5} seq")
        print(f"  val    {len(val_loader.dataset):>5} seq  (test/ folder)")

    test_loader = None
    if os.path.isdir(test_dir):
        _, test_loader = data_loader(args, {'root': test_dir, 'type': 'test'},
                                     test=True, test_year=args.test_year)
        print(f"  test   {len(test_loader.dataset):>5} seq  (year={args.test_year})\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len  = args.pred_len,
        obs_len   = args.obs_len,
        sigma_min = args.sigma_min,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters  {n_params:,}\n")

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer     = optim.AdamW(model.parameters(),
                                lr=args.g_learning_rate,
                                weight_decay=args.weight_decay)
    total_steps   = len(train_loader) * args.num_epochs
    warmup_steps  = len(train_loader) * args.warmup_epochs
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    saver         = BestModelSaver(patience=args.patience)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,ADE_km,FDE_km,"
                "12h_km,24h_km,48h_km,72h_km,"
                "fm,dir,step,disp,heading,smooth,pinn,"
                "epoch_s,ms_per_batch\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("  TRAINING")
    print("=" * 68)

    epoch_times: list[float] = []

    for epoch in range(args.num_epochs):
        model.train()

        sum_total = 0.0
        sum_parts = {k: 0.0 for k in ('fm','dir','step','disp','heading','smooth','pinn')}
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move_to(list(batch), device)
            bd = model.get_loss_breakdown(bl)

            optimizer.zero_grad()
            bd['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            sum_total += bd['total'].item()
            for k in sum_parts:
                sum_parts[k] += bd[k]

            if i % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  fm={bd['fm']:.3f}"
                      f"  heading={bd['heading']:.3f}"
                      f"  pinn={bd['pinn']:.4f}"
                      f"  lr={lr:.2e}")

        ep_s      = time.perf_counter() - t0
        epoch_times.append(ep_s)
        n         = len(train_loader)
        avg_train = sum_total / n
        avg_parts = {k: v / n for k, v in sum_parts.items()}

        # Validation loss (every epoch)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bl = move_to(list(batch), device)
                val_loss += model.get_loss(bl).item()
        avg_val = val_loss / len(val_loader)

        # Full evaluation (every val_freq epochs)
        if epoch % args.val_freq == 0 or epoch < 3:
            m = evaluate(model, val_loader, device,
                         num_ensemble=args.val_ensemble,
                         ddim_steps=args.ode_steps,
                         pred_len=args.pred_len)

            avg_ep = sum(epoch_times) / len(epoch_times)
            print(f"\n{'─'*68}")
            print(f"  Epoch {epoch:>3}  train={avg_train:.4f}  val={avg_val:.4f}"
                  f"  ({ep_s:.0f}s, avg {avg_ep:.0f}s)")
            print(f"  fm={avg_parts['fm']:.4f}  dir={avg_parts['dir']:.4f}"
                  f"  step={avg_parts['step']:.4f}  disp={avg_parts['disp']:.4f}")
            print(f"  heading={avg_parts['heading']:.4f}"
                  f"  smooth={avg_parts['smooth']:.4f}"
                  f"  pinn={avg_parts['pinn']:.4f}")
            print(f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
                  f"  24h={m.get('24h',0):.0f}"
                  f"  48h={m.get('48h',0):.0f}"
                  f"  72h={m.get('72h',0):.0f} km"
                  f"  ({m['ms_per_batch']:.0f}ms/batch)")

            # Milestone annotation
            for thr, tag in ((400,'📉'),(300,'🔵'),(200,'🎯'),(150,'🏆'),(100,'🌟')):
                if m['ADE'] < thr:
                    print(f"  {tag}  ADE < {thr} km")
                    break

            print(f"{'─'*68}\n")

            with open(log_path, 'a') as f:
                f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f"{m['ADE']:.1f},{m['FDE']:.1f},"
                        f"{m.get('12h',0):.1f},{m.get('24h',0):.1f},"
                        f"{m.get('48h',0):.1f},{m.get('72h',0):.1f},"
                        f"{avg_parts['fm']:.4f},{avg_parts['dir']:.4f},"
                        f"{avg_parts['step']:.4f},{avg_parts['disp']:.4f},"
                        f"{avg_parts['heading']:.4f},{avg_parts['smooth']:.4f},"
                        f"{avg_parts['pinn']:.4f},"
                        f"{ep_s:.1f},{m['ms_per_batch']:.1f}\n")

            saver(m['ADE'], model, args.output_dir, epoch,
                  optimizer, avg_train, avg_val)
        else:
            print(f"  Epoch {epoch:>3}  train={avg_train:.4f}  val={avg_val:.4f}"
                  f"  pinn={avg_parts['pinn']:.4f}  ({ep_s:.0f}s)")
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},"
                        f",,,,,,"
                        f"{avg_parts['fm']:.4f},{avg_parts['dir']:.4f},"
                        f"{avg_parts['step']:.4f},{avg_parts['disp']:.4f},"
                        f"{avg_parts['heading']:.4f},{avg_parts['smooth']:.4f},"
                        f"{avg_parts['pinn']:.4f},"
                        f"{ep_s:.1f},\n")

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.output_dir, f'ckpt_ep{epoch:03d}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()}, path)
            print(f"  💾  Checkpoint → {path}")

        if saver.early_stop:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # ── Final test ─────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  FINAL TEST  (held-out year={args.test_year})")
    print(f"{'='*68}")

    if test_loader:
        best_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded best model @ epoch {ckpt['epoch']}"
                  f"  (val ADE={ckpt['val_ade_km']:.1f} km)")

        tm = evaluate(model, test_loader, device,
                      num_ensemble=args.val_ensemble,
                      ddim_steps=args.ode_steps,
                      pred_len=args.pred_len)

        print(f"\n  ADE={tm['ADE']:.1f}  FDE={tm['FDE']:.1f}"
              f"  12h={tm.get('12h',0):.0f}"
              f"  24h={tm.get('24h',0):.0f}"
              f"  48h={tm.get('48h',0):.0f}"
              f"  72h={tm.get('72h',0):.0f} km")

        result_path = os.path.join(args.output_dir, 'test_results.txt')
        with open(result_path, 'w') as f:
            f.write(f"model         : TCFlowMatching v7\n")
            f.write(f"loss_eq       : Eq(60) FM=1.0 dir=2.0 step=0.5 disp=1.0 heading=2.0 smooth=0.2 pinn=0.5\n")
            f.write(f"pinn          : Full BVE with ERA5 u/v850 bilinear interp (fallback: simplified BVE)\n")
            f.write(f"sigma_min     : {args.sigma_min}\n")
            f.write(f"test_year     : {args.test_year}\n")
            f.write(f"ADE_km        : {tm['ADE']:.1f}\n")
            f.write(f"FDE_km        : {tm['FDE']:.1f}\n")
            for h in (12, 24, 48, 72):
                f.write(f"{h}h_km         : {tm.get(f'{h}h', 0):.1f}\n")
        print(f"\n  Results → {result_path}")

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
    print(f"  Avg epoch time : {avg_ep:.0f}s")
    print(f"  Log            : {log_path}")
    print(f"{'='*68}\n")


if __name__ == '__main__':
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)