"""
TCNM/losses.py
==============
Loss / metric utilities — updated for TC-Diffusion.

The diffusion model's sample() now returns:
  best_traj, best_Me, all_trajs, all_Me, scores, best_idx

This module's helper functions work with the [T, B, 2] tensor convention
used throughout the codebase (LONG = index 0, LAT = index 1).
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────
# Misc GAN-era losses (kept for compatibility)
# ─────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin=None):
        super().__init__()
        self.margin = margin
        self.Loss = (nn.SoftMarginLoss() if margin is None
                     else nn.TripletMarginLoss(margin=margin, p=2))

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            y = torch.ones(anchor.shape[0], device=anchor.device)
            return self.Loss(
                torch.norm(anchor - neg, 2, dim=1) -
                torch.norm(anchor - pos, 2, dim=1), y
            )
        return self.Loss(anchor, pos, neg)


def bce_loss(input, target):
    neg_abs = -input.abs()
    return (input.clamp(min=0) - input * target +
            (1 + neg_abs.exp()).log()).mean()


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    loss = (loss_mask.unsqueeze(2) *
            (pred_traj_gt.permute(1, 0, 2) -
             pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    return loss.sum(dim=2).sum(dim=1)


# ─────────────────────────────────────────────────
# De-normalisation
# ─────────────────────────────────────────────────

def toNE(pred_traj: torch.Tensor, pred_Me: torch.Tensor):
    """
    Denormalise model outputs → real physical units.

    Normalisation rules (from TCND paper, Table 1 / Equations 1-4):
        LONGnorm = (LONG − 1800) / 50   →  LONG = LONGnorm × 50 + 1800  (0.1 °E)
        LATnorm  = LAT / 50             →  LAT  = LATnorm  × 50          (0.1 °N)
        PRESnorm = (PRES − 960) / 50    →  PRES = PRESnorm × 50 + 960   (hPa)
        WNDnorm  = (WND  − 40)  / 25   →  WND  = WNDnorm  × 25 + 40    (m/s)

    Args
    ----
    pred_traj : [T, B, 2]  normalised [LONG, LAT]
    pred_Me   : [T, B, 2]  normalised [PRES, WND]

    Returns
    -------
    res_traj : [T, B, 2]  real [LONG (0.1°E), LAT (0.1°N)]
    res_me   : [T, B, 2]  real [PRES (hPa), WND (m/s)]
    """
    res_traj = pred_traj.clone()
    res_me   = pred_Me.clone()

    if res_traj.dim() == 2:
        res_traj = res_traj.unsqueeze(1)
        res_me   = res_me.unsqueeze(1)

    res_traj[:, :, 0] = res_traj[:, :, 0] * 50.0 + 1800.0   # LONG
    res_traj[:, :, 1] = res_traj[:, :, 1] * 50.0             # LAT

    res_me[:, :, 0]   = res_me[:, :, 0] * 50.0 + 960.0       # PRES
    res_me[:, :, 1]   = res_me[:, :, 1] * 25.0 + 40.0        # WND

    return res_traj, res_me


# ─────────────────────────────────────────────────
# Track error  (km)
# ─────────────────────────────────────────────────

def trajectory_displacement_error(
    pred_traj:    torch.Tensor,
    pred_traj_gt: torch.Tensor,
    mode:         str = 'sum',
) -> torch.Tensor:
    """
    Great-circle-approximate track error in kilometres.

    Uses the standard meteorological convention:
        Δlat_km  = Δlat_deg  × 111.0
        Δlon_km  = Δlon_deg  × 111.0 × cos(lat_gt)

    Args
    ----
    pred_traj    : [T, B, 2]  predicted  [LONG, LAT] in 0.1° units
    pred_traj_gt : [T, B, 2]  ground truth [LONG, LAT] in 0.1° units
    mode         : 'sum' | 'raw'
    """
    gt   = pred_traj_gt.permute(1, 0, 2)   # [B, T, 2]
    pred = pred_traj.permute(1, 0, 2)

    diff = gt - pred

    lon_diff_deg = diff[:, :, 0] / 10.0    # LONG difference (°)
    lat_diff_deg = diff[:, :, 1] / 10.0    # LAT  difference (°)

    lat_km = lat_diff_deg * 111.0
    lon_km = (lon_diff_deg * 111.0 *
              torch.cos(gt[:, :, 1] / 10.0 * torch.pi / 180.0))

    loss = torch.sqrt(lon_km ** 2 + lat_km ** 2)
    return torch.sum(loss) if mode == 'sum' else loss   # [B, T] when 'raw'


# ─────────────────────────────────────────────────
# Intensity error  (hPa / m/s)
# ─────────────────────────────────────────────────

def value_error(
    pred_traj:    torch.Tensor,
    pred_traj_gt: torch.Tensor,
    mode:         str = 'sum',
) -> torch.Tensor:
    """
    Mean absolute error for intensity (PRES hPa, WND m/s).

    Args
    ----
    pred_traj    : [T, B, 2]  predicted  [PRES, WND]
    pred_traj_gt : [T, B, 2]  ground truth
    mode         : 'sum' | 'raw'
    """
    loss = torch.abs(
        pred_traj.permute(1, 0, 2) - pred_traj_gt.permute(1, 0, 2)
    )                                       # [B, T, 2]
    return torch.sum(loss) if mode == 'sum' else loss


# ─────────────────────────────────────────────────
# Evaluation helper (works with diffusion sample())
# ─────────────────────────────────────────────────

def evaluate_diffusion_output(
    best_traj:    torch.Tensor,   # [T_pred, B, 2]  normalised
    best_Me:      torch.Tensor,   # [T_pred, B, 2]  normalised
    pred_traj_gt: torch.Tensor,   # [T_pred, B, 2]  normalised
    pred_Me_gt:   torch.Tensor,   # [T_pred, B, 2]  normalised
):
    """
    Convenience wrapper: de-normalise → compute TDE & VE in real units.

    Returns
    -------
    tde : [B, T_pred]   track distance error (km)
    ve  : [B, T_pred, 2] intensity absolute error ([hPa, m/s])
    """
    real_pred_traj, real_pred_Me = toNE(best_traj.clone(), best_Me.clone())
    real_gt_traj,   real_gt_Me   = toNE(pred_traj_gt.clone(), pred_Me_gt.clone())

    tde = trajectory_displacement_error(real_pred_traj, real_gt_traj, mode='raw')
    ve  = value_error(real_pred_Me, real_gt_Me, mode='raw')

    return tde, ve