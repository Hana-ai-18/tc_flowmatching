"""
scripts/visual_evaluate_model_Me.py - Fixed visualization
"""
import os, sys, random, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from TCNM.flow_matching_model import TCFlowMatching
from TCNM.data.loader import data_loader


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed fixed at {seed}")


def move_batch(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return tuple(out)


def detect_pred_len(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck.get('model_state', ck))
    key = 'denoiser.pos_encoding'
    if key in sd:
        return sd[key].shape[1]
    return 12


def denorm(norm_traj):
    """[N,2] norm → real 0.1° units"""
    r = np.zeros_like(norm_traj)
    r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0
    r[:, 1] = norm_traj[:, 1] * 50.0
    return r


def load_sat_image(him_path, year, name, timestamp):
    p = os.path.join(him_path, str(year), name.upper(), f"{timestamp}.png")
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # fallback: nearest file
    d = os.path.join(him_path, str(year), name.upper())
    if os.path.exists(d):
        pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        if pngs:
            tgt = datetime.strptime(timestamp, '%Y%m%d%H')
            best = min(pngs, key=lambda f: abs(
                (datetime.strptime(f[:-4], '%Y%m%d%H') - tgt).total_seconds()))
            img = cv2.imread(os.path.join(d, best))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.zeros((800, 800, 3), dtype=np.uint8)


def visualize_forecast(args):
    set_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pred_len = detect_pred_len(args.model_path)
    if args.pred_len != pred_len:
        print(f"⚠️  pred_len overridden: {args.pred_len} → {pred_len}")
        args.pred_len = pred_len

    # Load model
    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck    = torch.load(args.model_path, map_location=device, weights_only=False)
    sd    = ck.get('model_state_dict', ck.get('model_state', ck))
    model.load_state_dict(sd)
    model.eval()
    print("✅ Model loaded")

    # Load dataset
    dset, _ = data_loader(
        args, {'root': args.TC_data_path, 'type': args.dset_type},
        test=True, test_year=args.test_year)
    print(f"✅ {len(dset)} samples")

    t_name = args.tc_name.strip().upper()
    t_date = str(args.tc_date).strip()

    target = None
    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if (t_name in str(info['old'][1]).upper() and
                t_date == str(info['tydate'][args.obs_len])):
            target = item
            print(f"✅ Found: {info['old'][1]} @ {t_date}")
            break

    if target is None:
        print("❌ Not found! Check --tc_name and --tc_date")
        return

    # Build single-sample batch using seq_collate
    from TCNM.data.trajectoriesWithMe_unet_training import seq_collate
    batch = seq_collate([target])
    batch = move_batch(batch, device)

    with torch.no_grad():
        pred_traj_t, pred_Me_t = model.sample(batch)  # [T, B, 2]

    # Squeeze batch dim (B=1)
    obs_norm  = batch[0].squeeze(1).cpu().numpy()       # [obs_len, 2]
    gt_norm   = batch[1].squeeze(1).cpu().numpy()       # [pred_len, 2]
    pred_norm = pred_traj_t.squeeze(1).cpu().numpy()    # [pred_len, 2]

    obs_r  = denorm(obs_norm)
    gt_r   = denorm(gt_norm)
    pred_r = denorm(pred_norm)

    errors = np.linalg.norm(gt_r - pred_r, axis=1)
    print("\\nErrors (km):")
    for i, e in enumerate(errors):
        print(f"  +{(i+1)*6:3d}h: {e*11.1:6.1f} km")

    # ── Plot ──────────────────────────────────────────────────────────────────
    sat = load_sat_image(args.himawari_path, args.test_year, t_name, t_date)
    SZ  = 900
    sat = cv2.resize(sat, (SZ, SZ))

    ref = obs_r[-1]
    all_pts = np.vstack([obs_r, gt_r, pred_r])
    span = np.max(all_pts, 0) - np.min(all_pts, 0)
    scale = (SZ * 0.65) / (max(span) + 1e-6)

    def to_px(pts):
        dx = (pts[:, 0] - ref[0]) * scale + SZ/2
        dy = -(pts[:, 1] - ref[1]) * scale + SZ/2
        return dx, dy

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(sat, extent=[0, SZ, SZ, 0], alpha=0.55)

    ox, oy = to_px(obs_r)
    ax.plot(ox, oy, 'o-', color='#00FFFF', lw=3, ms=6,
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'Observed ({args.obs_len*6}h)', zorder=8)

    full_gt = np.vstack([ref.reshape(1,-1), gt_r])
    gx, gy  = to_px(full_gt)
    ax.plot(gx, gy, 'o-', color='#FF0000', lw=4, ms=8,
            markeredgecolor='white', markeredgewidth=2,
            label=f'Actual ({args.pred_len*6}h)', zorder=9)

    full_pred = np.vstack([ref.reshape(1,-1), pred_r])
    px, py = to_px(full_pred)
    ax.plot(px, py, 'o-', color='#00FF00', lw=4, ms=8,
            markeredgecolor='darkgreen', markeredgewidth=2,
            label=f'Forecast ({args.pred_len*6}h)', zorder=10)

    for step in [3, 7, 11]:
        if step < len(gx)-1 and step < len(errors):
            ax.plot([gx[step+1], px[step+1]], [gy[step+1], py[step+1]],
                    '--', color='yellow', lw=1.5, alpha=0.7, zorder=7)

    for i in range(len(px)):
        h = i * 6
        if i == 0:
            lbl, col = 'NOW', 'white'
        elif i % 2 == 0:
            ekm = errors[i-1]*11.1 if i > 0 else 0
            lbl, col = f'+{h}h\n{ekm:.0f}km', 'lime'
        else:
            continue
        ax.text(px[i], py[i]-25, lbl, fontsize=9, color=col, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='black', alpha=0.8, ec=col, lw=2),
                zorder=15)

    ax.scatter([SZ/2], [SZ/2], color='yellow', marker='*', s=700,
               edgecolors='red', linewidths=3, zorder=25, label='Current')

    title = (f"🌀 {t_name}  {args.pred_len*6}h FORECAST\\n"
             f"Mean: {np.mean(errors)*11.1:.0f} km  |  {args.pred_len*6}h: {errors[-1]*11.1:.0f} km")
    plt.title(title, fontsize=16, fontweight='bold', color='white', pad=15,
              bbox=dict(boxstyle='round,pad=0.8', fc='black', alpha=0.9, ec='cyan', lw=2))
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9,
               facecolor='black', edgecolor='cyan', labelcolor='white')
    plt.xlim(0, SZ); plt.ylim(SZ, 0); plt.axis('off')
    plt.tight_layout()

    out = f"forecast_{args.pred_len*6}h_{t_name}_{t_date}.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"\n✅ Saved: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',    required=True)
    p.add_argument('--TC_data_path',  required=True)
    p.add_argument('--himawari_path', required=True)
    p.add_argument('--tc_name',       default='WIPHA')
    p.add_argument('--tc_date',       default='2019073106')
    p.add_argument('--test_year',     type=int, default=2019)
    p.add_argument('--obs_len',       type=int, default=8)
    p.add_argument('--pred_len',      type=int, default=12)
    p.add_argument('--dset_type',     default='test')
    p.add_argument('--batch_size',    type=int, default=1)
    p.add_argument('--delim',         default=' ')
    p.add_argument('--skip',          type=int, default=1)
    p.add_argument('--min_ped',       type=int, default=1)
    p.add_argument('--threshold',     type=float, default=0.002)
    p.add_argument('--other_modal',   default='gph')
    visualize_forecast(p.parse_args())