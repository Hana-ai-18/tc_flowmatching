"""
TCNM/data/loader.py - Fixed data loader
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader
from TCNM.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


def _find_tcnd_root(path):
    """Smart path resolution to find TCND_vn root."""
    path = os.path.abspath(path)
    # Walk up until we find a dir that has Data1d inside
    check = path
    for _ in range(5):
        if os.path.exists(os.path.join(check, 'Data1d')):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent
    # Try appending TCND_vn
    candidate = os.path.join(path, 'TCND_vn')
    if os.path.exists(candidate):
        return candidate
    return path


def data_loader(args, path_config, test=False, test_year=None):
    if isinstance(path_config, dict):
        raw_path  = path_config.get('root', '')
        dset_type = path_config.get('type', 'test' if test else 'train')
    else:
        raw_path  = str(path_config)
        dset_type = 'test' if test else 'train'

    root = _find_tcnd_root(raw_path)

    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    dataset = TrajectoryDataset(
        data_dir   = root,
        obs_len    = args.obs_len,
        pred_len   = args.pred_len,
        skip       = args.skip,
        threshold  = args.threshold,
        min_ped    = args.min_ped,
        delim      = args.delim,
        other_modal= getattr(args, 'other_modal', 'gph'),
        test_year  = test_year,
        type       = dset_type,
        is_test    = test,
    )

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = not test,
        collate_fn  = seq_collate,
        num_workers = 0,
        drop_last   = False,
    )

    print(f"✅ {len(dataset)} sequences")
    return dataset, loader
'''


# ==============================================================
# FILE 5: scripts/train_diffusion.py
# ==============================================================

TRAIN_CODE = '''
"""
scripts/train_diffusion.py - Fixed training script
"""
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from TCNM.flow_matching_model import TCDiffusion
from TCNM.data.loader import data_loader


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device):
    """Move batch tensors to device."""
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return tuple(out)


def validate(model, loader, device):
    model.eval()
    total_ade, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred_traj, _ = model.sample(batch)
            gt_traj = batch[1]
            dist = torch.norm(pred_traj - gt_traj, dim=2)
            total_ade += dist.mean().item()
            n += 1
    avg = total_ade / max(n, 1)
    km  = avg * 50 * 11.1
    print(f"  Val ADE: {avg:.6f} | {km:.1f} km")
    return avg, km


def train_model(args):
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = TCDiffusion(
        pred_len  = args.pred_len,
        obs_len   = args.obs_len,
        num_steps = args.num_diffusion_steps,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ── Load data ────────────────────────────────────────────────────────────
    # FIX: don't do string replace; use a separate test path arg or same root
    train_dset, train_loader = data_loader(
        args, {'root': args.dataset_root, 'type': 'train'}, test=False)
    val_dset, val_loader = data_loader(
        args, {'root': args.dataset_root, 'type': 'test'}, test=True,
        test_year=args.val_year)

    print(f"Train: {len(train_dset)}  |  Val: {len(val_dset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.g_learning_rate,
                                   betas=(0.9,0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_km, patience_ctr = float('inf'), 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss, nb = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1

        print(f"Epoch {epoch:3d} | loss={epoch_loss/nb:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if epoch % 5 == 0:
            _, km = validate(model, val_loader, device)
            if km < best_km:
                best_km = km; patience_ctr = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'val_km': km, 'args': args},
                           os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  ✅ New best: {km:.1f} km")
                if km < 50:
                    print("  🎉 TARGET < 50 km ACHIEVED!")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print("Early stopping."); break

        if epoch % 20 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                       os.path.join(args.output_dir, f'ckpt_{epoch}.pth'))
        scheduler.step()

    print(f"\nDone. Best ADE: {best_km:.1f} km")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root',       required=True)
    p.add_argument('--output_dir',         default='./model_save')
    p.add_argument('--obs_len',            type=int, default=8)
    p.add_argument('--pred_len',           type=int, default=12)
    p.add_argument('--num_diffusion_steps',type=int, default=100)
    p.add_argument('--batch_size',         type=int, default=32)
    p.add_argument('--num_epochs',         type=int, default=300)
    p.add_argument('--g_learning_rate',    type=float, default=3e-4)
    p.add_argument('--grad_clip',          type=float, default=1.0)
    p.add_argument('--patience',           type=int, default=30)
    p.add_argument('--gpu_num',            type=int, default=0)
    p.add_argument('--val_year',           type=int, default=2019)
    p.add_argument('--delim',              default=' ')
    p.add_argument('--skip',               type=int, default=1)
    p.add_argument('--min_ped',            type=int, default=1)
    p.add_argument('--threshold',          type=float, default=0.002)
    p.add_argument('--other_modal',        default='gph')
    train_model(p.parse_args())