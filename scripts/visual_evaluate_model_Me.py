# """
# scripts/visual_evaluate_model_Me.py
# ====================================
# TC-Diffusion 72h Forecast Visualisation  ── FIXED VERSION

# FIXES:
# 1. Proper DDPM sampling (không dùng physics model đơn giản)
# 2. Coordinate system đúng: LONG → X (East=Right), LAT → Y (North=Up)
# 3. Anchor: trajectory bắt đầu từ current position
# 4. detect_pred_len() dùng đúng key 'denoiser.pos_enc'
# 5. Scale và offset tính lại đúng để track khớp satellite image
# """

# import os
# import sys
# import random
# import argparse

# import numpy as np
# import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import cv2
# from datetime import datetime

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)

# from TCNM.diffusion_model import TCDiffusion
# from TCNM.data.loader import data_loader
# from TCNM.data.trajectoriesWithMe_unet_training import seq_collate


# # ── Seed ─────────────────────────────────────────────────────────────────────

# def set_seed(s=42):
#     random.seed(s); np.random.seed(s)
#     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False
#     print(f"🔒 Seed fixed = {s}\n")


# # ── Device helpers ────────────────────────────────────────────────────────────

# def move_batch(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return tuple(out)


# # ── Checkpoint helpers ────────────────────────────────────────────────────────

# def detect_pred_len(ckpt_path):
#     """
#     Đọc pred_len từ pos_enc shape trong checkpoint.
#     KEY PHẢI KHỚP với DeterministicDenoiser: 'denoiser.pos_enc'
#     """
#     ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     sd = ck.get('model_state_dict', ck.get('model_state', ck))

#     # Thử đúng key trước
#     for key in ['denoiser.pos_enc', 'denoiser.pos_encoding']:
#         if key in sd:
#             return sd[key].shape[1]

#     # Fallback: tìm bất kỳ pos_enc nào
#     for k, v in sd.items():
#         if 'pos_enc' in k and v.dim() == 3:
#             return v.shape[1]

#     print("⚠️  Không tìm thấy pos_enc key, dùng pred_len=12")
#     return 12


# # ── Denormalization ────────────────────────────────────────────────────────────

# def denorm(norm_traj):
#     """
#     [N, 2] normalised → [N, 2] real (0.1° units)
#     LONG = norm * 50 + 1800  (0.1°E)
#     LAT  = norm * 50          (0.1°N)
#     """
#     r = np.zeros_like(norm_traj)
#     r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0   # LONG 0.1°E
#     r[:, 1] = norm_traj[:, 1] * 50.0             # LAT  0.1°N
#     return r


# def real_to_deg(pts_01):
#     """Convert 0.1° units → degrees"""
#     return pts_01 / 10.0


# # ── Pixel mapping ─────────────────────────────────────────────────────────────

# def to_pixels(coords_deg, ref_deg, ppu, cx, cy):
#     """
#     [N, 2] degrees (LONG, LAT) → screen pixels
#     LONG increases East → screen X increases right (+)
#     LAT  increases North → screen Y increases UP, but screen Y is inverted → (-)
#     """
#     d_long = coords_deg[:, 0] - ref_deg[0]   # dương = Đông
#     d_lat  = coords_deg[:, 1] - ref_deg[1]   # dương = Bắc

#     px = cx + d_long * ppu   # East  → right
#     py = cy - d_lat  * ppu   # North → up (Y inverted on screen)
#     return px, py


# # ── Satellite image ───────────────────────────────────────────────────────────

# def load_himawari(him_path, year, name, timestamp):
#     name  = name.strip().upper()
#     exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
#     if os.path.exists(exact):
#         img = cv2.imread(exact)
#         if img is not None:
#             print(f"🛰️  Loaded: {exact}")
#             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     d = os.path.join(him_path, str(year), name)
#     if os.path.exists(d):
#         pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
#         if pngs:
#             tgt  = datetime.strptime(timestamp, '%Y%m%d%H')
#             best = min(pngs, key=lambda f: abs(
#                 (datetime.strptime(f[:-4], '%Y%m%d%H') - tgt).total_seconds()))
#             path = os.path.join(d, best)
#             img  = cv2.imread(path)
#             if img is not None:
#                 print(f"🛰️  Nearest: {path}")
#                 return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     print("⚠️  Himawari image not found – black background")
#     return np.zeros((1000, 1000, 3), dtype=np.uint8)


# # ── Main ──────────────────────────────────────────────────────────────────────

# def visualize_forecast(args):
#     set_seed(42)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     print(f"{'='*65}")
#     print(f"  TC-Diffusion Forecast  |  {args.tc_name}  @  {args.tc_date}")
#     print(f"{'='*65}\n")

#     # ── Auto-detect pred_len ──────────────────────────────────────────────
#     detected = detect_pred_len(args.model_path)
#     if args.pred_len != detected:
#         print(f"⚠️  pred_len: {args.pred_len} → {detected} (from checkpoint)")
#         args.pred_len = detected

#     # ── Load model ────────────────────────────────────────────────────────
#     model = TCDiffusion(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck    = torch.load(args.model_path, map_location=device, weights_only=False)
#     sd    = ck.get('model_state_dict', ck.get('model_state', ck))
#     model.load_state_dict(sd)
#     model.eval()
#     print("✅ Model loaded\n")

#     # ── Load dataset ──────────────────────────────────────────────────────
#     dset, _ = data_loader(
#         args,
#         {'root': args.TC_data_path, 'type': args.dset_type},
#         test=True,
#         test_year=args.test_year,
#     )
#     print(f"✅ Dataset: {len(dset)} samples\n")

#     # ── Find typhoon sequence ─────────────────────────────────────────────
#     t_name  = args.tc_name.strip().upper()
#     t_date  = str(args.tc_date).strip()
#     target  = None

#     for i in range(len(dset)):
#         item = dset[i]
#         info = item[-1]
#         if (t_name in str(info['old'][1]).strip().upper()
#                 and t_date == str(info['tydate'][args.obs_len]).strip()):
#             target = item
#             print(f"✅ Found: {info['old'][1]}  @  {info['tydate'][args.obs_len]}\n")
#             break

#     if target is None:
#         print(f"❌ '{t_name} @ {t_date}' not found. Check --tc_name and --tc_date.")
#         # Gợi ý các sample có sẵn
#         print("Available samples (first 10):")
#         for i in range(min(10, len(dset))):
#             info = dset[i][-1]
#             print(f"  [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
#         return

#     # ── Build batch ───────────────────────────────────────────────────────
#     batch = move_batch(seq_collate([target]), device)

#     # ── Inference: proper DDPM sampling ──────────────────────────────────
#     print("🔄 Running DDPM reverse diffusion sampling...")
#     pred_traj_t, pred_Me_t = model.sample(batch)  # [T_pred, B, 2]
#     print("✅ Sampling done\n")

#     # ── Extract arrays ────────────────────────────────────────────────────
#     obs_n  = batch[0][:, 0, :].cpu().numpy()    # [T_obs,  2] normalized
#     gt_n   = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2] normalized
#     pred_n = pred_traj_t[:, 0, :].cpu().numpy() # [T_pred, 2] normalized

#     obs_r  = denorm(obs_n)   # [T_obs,  2] 0.1° units
#     gt_r   = denorm(gt_n)    # [T_pred, 2] 0.1° units
#     pred_r = denorm(pred_n)  # [T_pred, 2] 0.1° units

#     # Convert sang degrees để hiển thị
#     obs_deg  = real_to_deg(obs_r)   # [T_obs,  2] degrees
#     gt_deg   = real_to_deg(gt_r)    # [T_pred, 2] degrees
#     pred_deg = real_to_deg(pred_r)  # [T_pred, 2] degrees

#     # ── Error report ──────────────────────────────────────────────────────
#     # Error tính theo 0.1° units, 1 unit ≈ 11.1 km
#     errors_km = np.linalg.norm(gt_r - pred_r, axis=1) * 11.1

#     print("📊 Track errors:")
#     for i, e in enumerate(errors_km):
#         mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
#         print(f"   +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
#     print(f"\n   Mean : {errors_km.mean():6.1f} km")
#     n = len(errors_km)
#     if n >= 4:  print(f"   24h  : {errors_km[3]:6.1f} km")
#     if n >= 8:  print(f"   48h  : {errors_km[7]:6.1f} km")
#     if n >= 12: print(f"   72h  : {errors_km[11]:6.1f} km")
#     print()

#     # ── Coordinate ranges ─────────────────────────────────────────────────
#     ref_r   = obs_r[-1]                        # last observed (0.1° units)
#     ref_deg = real_to_deg(ref_r)               # degrees

#     print(f"📍 Current position: LONG={ref_deg[0]:.2f}°E, LAT={ref_deg[1]:.2f}°N")
#     print(f"   GT 72h: LONG={gt_deg[-1,0]:.2f}°E, LAT={gt_deg[-1,1]:.2f}°N")
#     print(f"   PD 72h: LONG={pred_deg[-1,0]:.2f}°E, LAT={pred_deg[-1,1]:.2f}°N\n")

#     # ── Satellite background ──────────────────────────────────────────────
#     sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)

#     SZ  = 1000
#     sat = cv2.resize(sat, (SZ, SZ))
#     cx, cy = SZ / 2.0, SZ / 2.0

#     # ── Scale: tính ppu (pixels per 0.1°) ────────────────────────────────
#     # Gom tất cả điểm để tính span
#     all_pts_deg = np.vstack([obs_deg, gt_deg, pred_deg])
#     span_long   = all_pts_deg[:, 0].max() - all_pts_deg[:, 0].min()
#     span_lat    = all_pts_deg[:, 1].max() - all_pts_deg[:, 1].min()
#     span_max    = max(span_long, span_lat, 5.0)  # tối thiểu 5°

#     # Dùng 65% canvas để hiển thị toàn bộ track
#     ppu = (SZ * 0.65) / span_max  # pixels per degree

#     # ── Convert to screen pixels ──────────────────────────────────────────
#     def px(pts_deg):
#         return to_pixels(pts_deg, ref_deg, ppu, cx, cy)

#     ox, oy  = px(obs_deg)

#     # Ground truth bắt đầu từ current position
#     gt_full  = np.vstack([ref_deg.reshape(1, -1), gt_deg])
#     gx, gy   = px(gt_full)

#     # Prediction bắt đầu từ current position
#     pred_full = np.vstack([ref_deg.reshape(1, -1), pred_deg])
#     bx, by    = px(pred_full)

#     # ── Plot ──────────────────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
#     ax.set_facecolor('black')
#     ax.imshow(sat, extent=[0, SZ, SZ, 0], alpha=0.60, zorder=0)

#     # 1. Observed (cyan)
#     ax.plot(ox, oy, 'o-', color='#00FFFF', linewidth=2.5, markersize=5,
#             markeredgecolor='white', markeredgewidth=1.2,
#             label=f'Observed ({args.obs_len * 6}h)', zorder=8, alpha=0.9)

#     # 2. Ground truth (đỏ)
#     ax.plot(gx, gy, 'o-', color='#FF2222', linewidth=4, markersize=8,
#             markeredgecolor='white', markeredgewidth=2,
#             label=f'Actual track ({args.pred_len * 6}h)', zorder=9, alpha=0.95)

#     # 3. Prediction (xanh lá)
#     ax.plot(bx, by, 'o-', color='#00FF44', linewidth=4, markersize=8,
#             markeredgecolor='#004400', markeredgewidth=2,
#             label=f'Predicted track ({args.pred_len * 6}h)', zorder=10, alpha=0.95)

#     # 4. Error connectors tại 24h / 48h / 72h
#     for step_idx, label_h in [(4, 24), (8, 48), (12, 72)]:
#         si = step_idx  # index trong full array (0=NOW, 1=+6h, ...)
#         if si < len(gx) and si < len(bx):
#             ax.plot([gx[si], bx[si]], [gy[si], by[si]],
#                     '--', color='#FFD700', linewidth=1.8, alpha=0.65, zorder=7)
#             # Ghi khoảng cách error
#             if step_idx - 1 < len(errors_km):
#                 mid_x = (gx[si] + bx[si]) / 2
#                 mid_y = (gy[si] + by[si]) / 2
#                 ax.text(mid_x, mid_y, f'{errors_km[step_idx-1]:.0f}km',
#                         fontsize=8, color='#FFD700', ha='center',
#                         bbox=dict(fc='black', alpha=0.6, ec='none', pad=1), zorder=18)

#     # 5. Time labels trên track xanh (prediction)
#     for i in range(len(bx)):
#         h = i * 6
#         if i == 0:
#             lbl, col, fs = 'NOW', 'white', 11
#         elif h % 12 == 0:
#             e_km = errors_km[i - 1] if i > 0 and i - 1 < len(errors_km) else 0
#             lbl  = f'+{h}h\n{e_km:.0f}km'
#             col  = '#AAFF66'
#             fs   = 9
#         else:
#             continue
#         ax.text(bx[i], by[i] - 28, lbl,
#                 fontsize=fs, color=col, ha='center', fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
#                           alpha=0.82, edgecolor=col, linewidth=1.5),
#                 zorder=16)

#     # 6. Hướng mũi tên cho track dự đoán (để thấy hướng di chuyển)
#     for i in range(0, len(bx) - 1, 2):
#         dx = bx[i+1] - bx[i]
#         dy = by[i+1] - by[i]
#         if abs(dx) + abs(dy) > 5:
#             ax.annotate('', xy=(bx[i+1], by[i+1]), xytext=(bx[i], by[i]),
#                        arrowprops=dict(arrowstyle='->', color='#00FF44',
#                                       lw=1.5, mutation_scale=15),
#                        zorder=11)

#     # 7. Current position ★
#     ax.scatter([cx], [cy], color='#FFD700', marker='*', s=900,
#                edgecolors='#FF4400', linewidths=3, zorder=25,
#                label='Current position')

#     # ── Title ─────────────────────────────────────────────────────────────
#     dt_str = datetime.strptime(t_date, '%Y%m%d%H').strftime('%d %b %Y  %H:%M UTC')
#     fh     = args.pred_len * 6
#     mean_e = errors_km.mean()
#     last_e = errors_km[-1]

#     ax.set_title(
#         f"🌀  {t_name}  –  {fh}h TC-Diffusion Forecast\n"
#         f"📅  {dt_str}    │    Mean: {mean_e:.0f} km    │    {fh}h: {last_e:.0f} km",
#         fontsize=17, fontweight='bold', color='white', pad=18,
#         bbox=dict(boxstyle='round,pad=0.9', facecolor='#000000',
#                   alpha=0.92, edgecolor='#00FFFF', linewidth=2.5),
#     )

#     # ── Legend ────────────────────────────────────────────────────────────
#     ax.legend(loc='upper right', fontsize=12, framealpha=0.92,
#               facecolor='#111111', edgecolor='#00FFFF',
#               labelcolor='white', title='Track Legend',
#               title_fontsize=13)

#     # ── Info panel (lower-left) ───────────────────────────────────────────
#     lines = [
#         "Model : TC-Diffusion (DDPM)",
#         f"Obs   : {args.obs_len} × 6h = {args.obs_len*6}h",
#         f"Pred  : {args.pred_len} × 6h = {fh}h",
#         f"Ref   : {ref_deg[0]:.1f}°E  {ref_deg[1]:.1f}°N",
#         "",
#         "Track Errors (km):",
#     ]
#     for i, e in enumerate(errors_km):
#         h = (i + 1) * 6
#         if h in [12, 24, 48, 72] and h <= fh:
#             lines.append(f"  {h:3d}h : {e:6.1f}")
#     lines.append(f"  Mean : {mean_e:6.1f}")

#     ax.text(0.02, 0.02, '\n'.join(lines),
#             transform=ax.transAxes, fontsize=10, va='bottom',
#             family='monospace', color='#88FF88',
#             bbox=dict(boxstyle='round,pad=0.6', facecolor='black',
#                       alpha=0.88, edgecolor='white', linewidth=1.5),
#             zorder=20)

#     # ── Compass rose (nhỏ, góc dưới phải) ────────────────────────────────
#     ax.annotate('N', xy=(0.96, 0.12), xytext=(0.96, 0.08),
#                 xycoords='axes fraction',
#                 fontsize=12, color='white', ha='center', fontweight='bold',
#                 arrowprops=dict(arrowstyle='->', color='white', lw=2),
#                 zorder=30)

#     ax.set_xlim(0, SZ); ax.set_ylim(SZ, 0)
#     ax.axis('off')
#     plt.tight_layout()

#     # ── Save ──────────────────────────────────────────────────────────────
#     out = f"forecast_{fh}h_{t_name}_{t_date}.png"
#     plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
#     plt.close()
#     print(f"✅ Saved → {out}\n")


# # ── CLI ───────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     p = argparse.ArgumentParser(description='TC-Diffusion Forecast Visualisation (FIXED)')
#     p.add_argument('--model_path',    required=True,  help='Path to best_model.pth')
#     p.add_argument('--TC_data_path',  required=True,  help='TCND_vn root directory')
#     p.add_argument('--himawari_path', required=True,  help='Himawari image directory')
#     p.add_argument('--tc_name',       default='WIPHA')
#     p.add_argument('--tc_date',       default='2019073106',
#                    help='Thời điểm bắt đầu dự báo (obs_len cuối = thời điểm này)')
#     p.add_argument('--test_year',     type=int,   default=2019)
#     p.add_argument('--obs_len',       type=int,   default=8)
#     p.add_argument('--pred_len',      type=int,   default=12,
#                    help='Tự động detect từ checkpoint nếu khác')
#     p.add_argument('--dset_type',     default='test')
#     p.add_argument('--batch_size',    type=int,   default=1)
#     p.add_argument('--delim',         default=' ')
#     p.add_argument('--skip',          type=int,   default=1)
#     p.add_argument('--min_ped',       type=int,   default=1)
#     p.add_argument('--threshold',     type=float, default=0.002)
#     p.add_argument('--other_modal',   default='gph')
#     visualize_forecast(p.parse_args())

# __________________ new version: scripts/visual_evaluate_model_Me.py ____
"""
scripts/visual_evaluate_model_Me.py
====================================
TC-FlowMatching 72h Forecast Visualisation  ── FIXED VERSION

FIXES:
1. Proper DDPM sampling (không dùng physics model đơn giản)
2. Coordinate system đúng: LONG → X (East=Right), LAT → Y (North=Up)
3. Anchor: trajectory bắt đầu từ current position
4. detect_pred_len() dùng đúng key 'denoiser.pos_enc'
5. Scale và offset tính lại đúng để track khớp satellite image
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import matplotlib


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from TCNM.flow_matching_model import TCFlowMatching
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


from TCNM.data.loader import data_loader
from TCNM.data.trajectoriesWithMe_unet_training import seq_collate


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"🔒 Seed fixed = {s}\n")


# ── Device helpers ────────────────────────────────────────────────────────────

def move_batch(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return tuple(out)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def detect_pred_len(ckpt_path):
    """
    Đọc pred_len từ pos_enc shape trong checkpoint.
    KEY PHẢI KHỚP với DeterministicDenoiser: 'denoiser.pos_enc'
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck.get('model_state', ck))

    # Thử đúng key trước
    for key in ['denoiser.pos_enc', 'denoiser.pos_encoding']:
        if key in sd:
            return sd[key].shape[1]

    # Fallback: tìm bất kỳ pos_enc nào
    for k, v in sd.items():
        if 'pos_enc' in k and v.dim() == 3:
            return v.shape[1]

    print("⚠️  Không tìm thấy pos_enc key, dùng pred_len=12")
    return 12


# ── Denormalization ────────────────────────────────────────────────────────────

def denorm(norm_traj):
    """
    [N, 2] normalised → [N, 2] real (0.1° units)
    LONG = norm * 50 + 1800  (0.1°E)
    LAT  = norm * 50          (0.1°N)
    """
    r = np.zeros_like(norm_traj)
    r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0   # LONG 0.1°E
    r[:, 1] = norm_traj[:, 1] * 50.0             # LAT  0.1°N
    return r


def real_to_deg(pts_01):
    """Convert 0.1° units → degrees"""
    return pts_01 / 10.0


# ── Pixel mapping ─────────────────────────────────────────────────────────────

def to_pixels(coords_deg, ref_deg, ppu, cx, cy):
    """
    [N, 2] degrees (LONG, LAT) → screen pixels
    LONG increases East → screen X increases right (+)
    LAT  increases North → screen Y increases UP, but screen Y is inverted → (-)
    """
    d_long = coords_deg[:, 0] - ref_deg[0]   # dương = Đông
    d_lat  = coords_deg[:, 1] - ref_deg[1]   # dương = Bắc

    px = cx + d_long * ppu   # East  → right
    py = cy - d_lat  * ppu   # North → up (Y inverted on screen)
    return px, py


# ── Satellite image ───────────────────────────────────────────────────────────

def load_himawari(him_path, year, name, timestamp):
    name  = name.strip().upper()
    exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
    if os.path.exists(exact):
        img = cv2.imread(exact)
        if img is not None:
            print(f"🛰️  Loaded: {exact}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    d = os.path.join(him_path, str(year), name)
    if os.path.exists(d):
        pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        if pngs:
            tgt  = datetime.strptime(timestamp, '%Y%m%d%H')
            best = min(pngs, key=lambda f: abs(
                (datetime.strptime(f[:-4], '%Y%m%d%H') - tgt).total_seconds()))
            path = os.path.join(d, best)
            img  = cv2.imread(path)
            if img is not None:
                print(f"🛰️  Nearest: {path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("⚠️  Himawari image not found – black background")
    return np.zeros((1000, 1000, 3), dtype=np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize_forecast(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*65}")
    print(f"  TC-FlowMatching Forecast  |  {args.tc_name}  @  {args.tc_date}")
    print(f"{'='*65}\n")

    # ── Auto-detect pred_len ──────────────────────────────────────────────
    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"⚠️  pred_len: {args.pred_len} → {detected} (from checkpoint)")
        args.pred_len = detected

    # ── Load model ────────────────────────────────────────────────────────
    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck    = torch.load(args.model_path, map_location=device, weights_only=False)
    sd      = ck.get('model_state_dict', ck.get('model_state', ck))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"⚠️  Missing keys (new layers, random init): {len(missing)}")
    if unexpected:
        print(f"⚠️  Unexpected keys (old layers, ignored): {len(unexpected)}")
    model.eval()
    print("✅ Model loaded\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    dset, _ = data_loader(
        args,
        {'root': args.TC_data_path, 'type': args.dset_type},
        test=True,
        test_year=args.test_year,
    )
    print(f"✅ Dataset: {len(dset)} samples\n")

    # ── Find typhoon sequence ─────────────────────────────────────────────
    t_name  = args.tc_name.strip().upper()
    t_date  = str(args.tc_date).strip()
    target  = None

    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if (t_name in str(info['old'][1]).strip().upper()
                and t_date == str(info['tydate'][args.obs_len]).strip()):
            target = item
            print(f"✅ Found: {info['old'][1]}  @  {info['tydate'][args.obs_len]}\n")
            break

    if target is None:
        print(f"❌ '{t_name} @ {t_date}' not found. Check --tc_name and --tc_date.")
        # Gợi ý các sample có sẵn
        print("Available samples (first 10):")
        for i in range(min(10, len(dset))):
            info = dset[i][-1]
            print(f"  [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
        return

    # ── Build batch ───────────────────────────────────────────────────────
    batch = move_batch(seq_collate([target]), device)

    # ── Inference: proper DDPM sampling ──────────────────────────────────
    print("🔄 Running DDPM reverse diffusion sampling...")
    pred_traj_t, pred_Me_t = model.sample(batch)  # [T_pred, B, 2]
    print("✅ Sampling done\n")

    # ── Extract arrays ────────────────────────────────────────────────────
    obs_n  = batch[0][:, 0, :].cpu().numpy()    # [T_obs,  2] normalized
    gt_n   = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2] normalized
    pred_n = pred_traj_t[:, 0, :].cpu().numpy() # [T_pred, 2] normalized

    obs_r  = denorm(obs_n)   # [T_obs,  2] 0.1° units
    gt_r   = denorm(gt_n)    # [T_pred, 2] 0.1° units
    pred_r = denorm(pred_n)  # [T_pred, 2] 0.1° units

    # Convert sang degrees để hiển thị
    obs_deg  = real_to_deg(obs_r)   # [T_obs,  2] degrees
    gt_deg   = real_to_deg(gt_r)    # [T_pred, 2] degrees
    pred_deg = real_to_deg(pred_r)  # [T_pred, 2] degrees

    # ── Error report ──────────────────────────────────────────────────────
    # Error tính theo 0.1° units, 1 unit ≈ 11.1 km
    errors_km = np.linalg.norm(gt_r - pred_r, axis=1) * 11.1

    print("📊 Track errors:")
    for i, e in enumerate(errors_km):
        mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
        print(f"   +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
    print(f"\n   Mean : {errors_km.mean():6.1f} km")
    n = len(errors_km)
    if n >= 4:  print(f"   24h  : {errors_km[3]:6.1f} km")
    if n >= 8:  print(f"   48h  : {errors_km[7]:6.1f} km")
    if n >= 12: print(f"   72h  : {errors_km[11]:6.1f} km")
    print()

    # ── Coordinate ranges ─────────────────────────────────────────────────
    ref_r   = obs_r[-1]                        # last observed (0.1° units)
    ref_deg = real_to_deg(ref_r)               # degrees

    print(f"📍 Current position: LONG={ref_deg[0]:.2f}°E, LAT={ref_deg[1]:.2f}°N")
    print(f"   GT 72h: LONG={gt_deg[-1,0]:.2f}°E, LAT={gt_deg[-1,1]:.2f}°N")
    print(f"   PD 72h: LONG={pred_deg[-1,0]:.2f}°E, LAT={pred_deg[-1,1]:.2f}°N\n")

    # ── Satellite background ──────────────────────────────────────────────
    sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)

    SZ  = 1000
    sat = cv2.resize(sat, (SZ, SZ))
    cx, cy = SZ / 2.0, SZ / 2.0

    # ── Scale: tính ppu (pixels per 0.1°) ────────────────────────────────
    # Gom tất cả điểm để tính span
    all_pts_deg = np.vstack([obs_deg, gt_deg, pred_deg])
    span_long   = all_pts_deg[:, 0].max() - all_pts_deg[:, 0].min()
    span_lat    = all_pts_deg[:, 1].max() - all_pts_deg[:, 1].min()
    span_max    = max(span_long, span_lat, 5.0)  # tối thiểu 5°

    # Dùng 65% canvas để hiển thị toàn bộ track
    ppu = (SZ * 0.65) / span_max  # pixels per degree

    # ── Convert to screen pixels ──────────────────────────────────────────
    def px(pts_deg):
        return to_pixels(pts_deg, ref_deg, ppu, cx, cy)

    ox, oy  = px(obs_deg)

    # Ground truth bắt đầu từ current position
    gt_full  = np.vstack([ref_deg.reshape(1, -1), gt_deg])
    gx, gy   = px(gt_full)

    # Prediction bắt đầu từ current position
    pred_full = np.vstack([ref_deg.reshape(1, -1), pred_deg])
    bx, by    = px(pred_full)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
    ax.set_facecolor('black')
    ax.imshow(sat, extent=[0, SZ, SZ, 0], alpha=0.60, zorder=0)

    # 1. Observed (cyan)
    ax.plot(ox, oy, 'o-', color='#00FFFF', linewidth=2.5, markersize=5,
            markeredgecolor='white', markeredgewidth=1.2,
            label=f'Observed ({args.obs_len * 6}h)', zorder=8, alpha=0.9)

    # 2. Ground truth (đỏ)
    ax.plot(gx, gy, 'o-', color='#FF2222', linewidth=4, markersize=8,
            markeredgecolor='white', markeredgewidth=2,
            label=f'Actual track ({args.pred_len * 6}h)', zorder=9, alpha=0.95)

    # 3. Prediction (xanh lá)
    ax.plot(bx, by, 'o-', color='#00FF44', linewidth=4, markersize=8,
            markeredgecolor='#004400', markeredgewidth=2,
            label=f'Predicted track ({args.pred_len * 6}h)', zorder=10, alpha=0.95)

    # 4. Error connectors tại 24h / 48h / 72h
    for step_idx, label_h in [(4, 24), (8, 48), (12, 72)]:
        si = step_idx  # index trong full array (0=NOW, 1=+6h, ...)
        if si < len(gx) and si < len(bx):
            ax.plot([gx[si], bx[si]], [gy[si], by[si]],
                    '--', color='#FFD700', linewidth=1.8, alpha=0.65, zorder=7)
            # Ghi khoảng cách error
            if step_idx - 1 < len(errors_km):
                mid_x = (gx[si] + bx[si]) / 2
                mid_y = (gy[si] + by[si]) / 2
                ax.text(mid_x, mid_y, f'{errors_km[step_idx-1]:.0f}km',
                        fontsize=8, color='#FFD700', ha='center',
                        bbox=dict(fc='black', alpha=0.6, ec='none', pad=1), zorder=18)

    # 5. Time labels trên track xanh (prediction)
    for i in range(len(bx)):
        h = i * 6
        if i == 0:
            lbl, col, fs = 'NOW', 'white', 11
        elif h % 12 == 0:
            e_km = errors_km[i - 1] if i > 0 and i - 1 < len(errors_km) else 0
            lbl  = f'+{h}h\n{e_km:.0f}km'
            col  = '#AAFF66'
            fs   = 9
        else:
            continue
        ax.text(bx[i], by[i] - 28, lbl,
                fontsize=fs, color=col, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                          alpha=0.82, edgecolor=col, linewidth=1.5),
                zorder=16)

    # 6. Hướng mũi tên cho track dự đoán (để thấy hướng di chuyển)
    for i in range(0, len(bx) - 1, 2):
        dx = bx[i+1] - bx[i]
        dy = by[i+1] - by[i]
        if abs(dx) + abs(dy) > 5:
            ax.annotate('', xy=(bx[i+1], by[i+1]), xytext=(bx[i], by[i]),
                       arrowprops=dict(arrowstyle='->', color='#00FF44',
                                      lw=1.5, mutation_scale=15),
                       zorder=11)

    # 7. Current position ★
    ax.scatter([cx], [cy], color='#FFD700', marker='*', s=900,
               edgecolors='#FF4400', linewidths=3, zorder=25,
               label='Current position')

    # ── Title ─────────────────────────────────────────────────────────────
    dt_str = datetime.strptime(t_date, '%Y%m%d%H').strftime('%d %b %Y  %H:%M UTC')
    fh     = args.pred_len * 6
    mean_e = errors_km.mean()
    last_e = errors_km[-1]

    ax.set_title(
        f"🌀  {t_name}  –  {fh}h TC-FlowMatching Forecast\n"
        f"📅  {dt_str}    │    Mean: {mean_e:.0f} km    │    {fh}h: {last_e:.0f} km",
        fontsize=17, fontweight='bold', color='white', pad=18,
        bbox=dict(boxstyle='round,pad=0.9', facecolor='#000000',
                  alpha=0.92, edgecolor='#00FFFF', linewidth=2.5),
    )

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(loc='upper right', fontsize=12, framealpha=0.92,
              facecolor='#111111', edgecolor='#00FFFF',
              labelcolor='white', title='Track Legend',
              title_fontsize=13)

    # ── Info panel (lower-left) ───────────────────────────────────────────
    lines = [
        "Model : TC-FlowMatching (DDPM)",
        f"Obs   : {args.obs_len} × 6h = {args.obs_len*6}h",
        f"Pred  : {args.pred_len} × 6h = {fh}h",
        f"Ref   : {ref_deg[0]:.1f}°E  {ref_deg[1]:.1f}°N",
        "",
        "Track Errors (km):",
    ]
    for i, e in enumerate(errors_km):
        h = (i + 1) * 6
        if h in [12, 24, 48, 72] and h <= fh:
            lines.append(f"  {h:3d}h : {e:6.1f}")
    lines.append(f"  Mean : {mean_e:6.1f}")

    ax.text(0.02, 0.02, '\n'.join(lines),
            transform=ax.transAxes, fontsize=10, va='bottom',
            family='monospace', color='#88FF88',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='black',
                      alpha=0.88, edgecolor='white', linewidth=1.5),
            zorder=20)

    # ── Compass rose (nhỏ, góc dưới phải) ────────────────────────────────
    ax.annotate('N', xy=(0.96, 0.12), xytext=(0.96, 0.08),
                xycoords='axes fraction',
                fontsize=12, color='white', ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                zorder=30)

    ax.set_xlim(0, SZ); ax.set_ylim(SZ, 0)
    ax.axis('off')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    out = f"forecast_{fh}h_{t_name}_{t_date}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"✅ Saved → {out}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TC-FlowMatching Forecast Visualisation (FIXED)')
    p.add_argument('--model_path',    required=True,  help='Path to best_model.pth')
    p.add_argument('--TC_data_path',  required=True,  help='TCND_vn root directory')
    p.add_argument('--himawari_path', required=True,  help='Himawari image directory')
    p.add_argument('--tc_name',       default='WIPHA')
    p.add_argument('--tc_date',       default='2019073106',
                   help='Thời điểm bắt đầu dự báo (obs_len cuối = thời điểm này)')
    p.add_argument('--test_year',     type=int,   default=2019)
    p.add_argument('--obs_len',       type=int,   default=8)
    p.add_argument('--pred_len',      type=int,   default=12,
                   help='Tự động detect từ checkpoint nếu khác')
    p.add_argument('--dset_type',     default='test')
    p.add_argument('--batch_size',    type=int,   default=1)
    p.add_argument('--delim',         default=' ')
    p.add_argument('--skip',          type=int,   default=1)
    p.add_argument('--min_ped',       type=int,   default=1)
    p.add_argument('--threshold',     type=float, default=0.002)
    p.add_argument('--other_modal',   default='gph')
    visualize_forecast(p.parse_args())