"""
utils/metrics.py  ── v2
========================
TC Track Forecasting Metrics — 3-tier evaluation framework.

Tier 1 — Baseline:
    ADE, FDE (Haversine-based, corrected for latitude)

Tier 2 — Stratified:
    ADE_str, ADE_rec (split by total rotation Θ)
    Δ_rec = ADE_rec - ADE_str  (straight-track bias indicator)

Tier 3 — Geometric:
    DTW  (Dynamic Time Warping with Haversine local cost)
    HE₄  (Heading Error at 24h)
    HE₈  (Heading Error at 48h)
    HE₁₂ (Heading Error at 72h)

References:
    Haversine: Sinnott (1984)
    DTW: Sakoe & Chiba (1978)
    BVE / Barotropic Vorticity: Holton & Hakim (2012)

TCND_VN normalization convention:
    norm_lon = (lon_01E − 1800) / 50   →   lon_01E = norm × 50 + 1800
    norm_lat = lat_01N / 50            →   lat_01N = norm × 50
    (0.1° units — divide by 10 for degrees)
"""

from __future__ import annotations

import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
R_EARTH_KM = 6371.0        # mean Earth radius
STEP_HOURS = 6             # hours per prediction step
PRED_LEN   = 12            # default prediction length (12 steps = 72h)

# Horizon → 0-based step index
HORIZON_STEPS: Dict[int, int] = {
    12:  1,    # step index 1 (second step, since step 0 = +6h)
    24:  3,
    36:  5,
    48:  7,
    60:  9,
    72: 11,
}

RECURVATURE_THRESHOLD_DEG = 45.0   # total rotation angle threshold


# ══════════════════════════════════════════════════════════════════════════════
#  Primitive: Haversine distance
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km_np(
    p1: np.ndarray,
    p2: np.ndarray,
    lon_idx: int = 0,
    lat_idx: int = 1,
    unit_01deg: bool = True,
) -> np.ndarray:
    """
    Haversine distance (km) between arrays of positions.

    Args:
        p1, p2   : shape [..., 2+]
        lon_idx  : channel index for longitude
        lat_idx  : channel index for latitude
        unit_01deg: if True, values are in 0.1° units (TCND_VN convention)
                    → divide by 10 for degrees

    Formula:
        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        d = 2·R·arcsin(√a)

    Why not Euclidean × 11.1:
        1° longitude ≈ 111·cos(lat) km
        At lat=25°N: error ~10% per step → ~48 km bias at 72h
        Different storms at different latitudes → inconsistent ADE
    """
    scale = 10.0 if unit_01deg else 1.0

    lat1 = np.deg2rad(p1[..., lat_idx] / scale)
    lat2 = np.deg2rad(p2[..., lat_idx] / scale)
    lon1 = np.deg2rad(p1[..., lon_idx] / scale)
    lon2 = np.deg2rad(p2[..., lon_idx] / scale)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2)
         * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)

    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def haversine_km_torch(
    pred,
    gt,
    lon_idx: int = 0,
    lat_idx: int = 1,
    unit_01deg: bool = True,
) :
    """Haversine distance (km) — PyTorch version. Returns torch.Tensor."""
    scale = 10.0 if unit_01deg else 1.0

    lat1 = torch.deg2rad(gt[..., lat_idx]   / scale)
    lat2 = torch.deg2rad(pred[..., lat_idx] / scale)
    lon1 = torch.deg2rad(gt[..., lon_idx]   / scale)
    lon2 = torch.deg2rad(pred[..., lon_idx] / scale)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (torch.sin(dlat / 2.0) ** 2
         + torch.cos(lat1) * torch.cos(lat2)
         * torch.sin(dlon / 2.0) ** 2)
    a = a.clamp(0.0, 1.0)

    return 2.0 * R_EARTH_KM * torch.asin(torch.sqrt(a))


def denorm_traj_np(n: np.ndarray) -> np.ndarray:
    """Normalized → 0.1° units (NumPy)."""
    r = n.copy()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def denorm_traj_torch(n):
    """Normalized → 0.1° units (PyTorch)."""
    r = n.clone()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 1 — Baseline: ADE & FDE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ade_fde(
    pred_01deg: np.ndarray,
    gt_01deg:   np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    ADE and FDE from denormed trajectories.

    Args:
        pred_01deg, gt_01deg: shape [T, 2+] — single sequence, in 0.1° units

    Returns:
        ade         : float (km)
        fde         : float (km)
        per_step_km : np.ndarray [T] — error at each step
    """
    per_step = haversine_km_np(pred_01deg, gt_01deg)   # [T]
    return float(per_step.mean()), float(per_step[-1]), per_step


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 2 — Stratified: straight-track vs recurvature
# ══════════════════════════════════════════════════════════════════════════════

def total_rotation_angle(
    traj_01deg: np.ndarray,
    lon_idx: int = 0,
    lat_idx: int = 1,
) -> float:
    """
    Compute total rotation angle Θ of a trajectory (degrees).

    Θ = Σ arccos( v_{k+1} · v_{k} / (|v_{k+1}| |v_{k}|) )

    Uses lat/lon-corrected velocity vectors (approximate Cartesian):
        u_k = (Δlon_k) × cos(lat_k)   [east component, corrected]
        v_k = (Δlat_k)                 [north component]

    Classification:
        Θ < 45°  → straight-track
        Θ ≥ 45°  → recurvature

    Why this formula:
        Pure lon/lat differences give wrong dot products at high latitudes
        because lon spacing is compressed. Cosine correction makes the
        velocity vectors physically meaningful.
    """
    T = traj_01deg.shape[0]
    if T < 3:
        return 0.0

    # Lat-corrected velocity vectors [T-1, 2]
    lats_rad = np.deg2rad(traj_01deg[:, lat_idx] / 10.0)
    cos_lat  = np.cos(lats_rad[:-1])                  # [T-1]

    dlat = np.diff(traj_01deg[:, lat_idx])             # [T-1]
    dlon = np.diff(traj_01deg[:, lon_idx]) * cos_lat   # [T-1], corrected

    v = np.stack([dlon, dlat], axis=-1)   # [T-1, 2]

    # Cumulative angle between consecutive velocity vectors
    total = 0.0
    for i in range(len(v) - 1):
        v1, v2 = v[i], v[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total += np.degrees(np.arccos(cos_a))

    return total


def classify_trajectory(
    gt_01deg: np.ndarray,
    threshold_deg: float = RECURVATURE_THRESHOLD_DEG,
) -> str:
    """Returns 'straight' or 'recurvature'."""
    theta = total_rotation_angle(gt_01deg)
    return 'recurvature' if theta >= threshold_deg else 'straight'


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 3a — DTW (Dynamic Time Warping)
# ══════════════════════════════════════════════════════════════════════════════

def dtw_haversine(
    s: np.ndarray,
    t: np.ndarray,
) -> float:
    """
    DTW distance between two trajectories using Haversine local cost.

    Args:
        s, t: shape [T, 2+] — in 0.1° units

    Returns:
        dtw_km: float — accumulated DTW cost (km)

    Why DTW over Fréchet:
        Fréchet uses monotone matching — penalizes heavily if recurvature
        is predicted correctly but shifted by 6h (one step early/late).
        DTW allows time-elastic matching: correct pattern but shifted
        timing still scores well.

    Time complexity: O(T²) — manageable for T=12 (12×12 = 144 ops)
    """
    n, m = len(s), len(t)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(haversine_km_np(
                s[i - 1:i], t[j - 1:j]
            ).item())
            dp[i, j] = cost + min(
                dp[i - 1, j],      # insertion
                dp[i, j - 1],      # deletion
                dp[i - 1, j - 1],  # match
            )

    return float(dp[n, m])


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 3b — Heading Error (HE)
# ══════════════════════════════════════════════════════════════════════════════

def heading_error_deg(
    pred_01deg: np.ndarray,
    gt_01deg:   np.ndarray,
    step_idx:   int,
    lon_idx:    int = 0,
    lat_idx:    int = 1,
) -> float:
    """
    Heading Error at a specific forecast step (degrees).

    HE_k = arccos( v̂_pred · v̂_gt ) ∈ [0°, 180°]

    where v̂ = lat-corrected velocity unit vector at step k.

    Args:
        pred_01deg, gt_01deg: [T, 2+] in 0.1° units
        step_idx: 0-based step index (e.g., 3 for 24h, 7 for 48h, 11 for 72h)

    Returns:
        heading_error: float (degrees, ∈ [0, 180])

    Why not use FDE:
        HE₁₂ = 90° distinguishes "storm hit Central Vietnam instead of
        Northern Vietnam" — FDE = 200 km cannot capture this.
        Two storms with same FDE can have very different heading errors.
    """
    T = pred_01deg.shape[0]

    # Need step_idx and step_idx+1 for velocity
    if step_idx < 0 or step_idx + 1 >= T:
        return float('nan')

    # Lat-corrected velocity vectors
    def _vel(traj, k):
        lat_k   = np.deg2rad(traj[k, lat_idx] / 10.0)
        cos_lat = np.cos(lat_k)
        dlat = traj[k + 1, lat_idx] - traj[k, lat_idx]
        dlon = (traj[k + 1, lon_idx] - traj[k, lon_idx]) * cos_lat
        return np.array([dlon, dlat], dtype=np.float64)

    v_pred = _vel(pred_01deg, step_idx)
    v_gt   = _vel(gt_01deg,   step_idx)

    n_pred = np.linalg.norm(v_pred)
    n_gt   = np.linalg.norm(v_gt)

    if n_pred < 1e-8 or n_gt < 1e-8:
        return float('nan')

    cos_a = np.clip(
        np.dot(v_pred, v_gt) / (n_pred * n_gt),
        -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cos_a)))


def compute_heading_errors(
    pred_01deg: np.ndarray,
    gt_01deg:   np.ndarray,
) -> Dict[str, float]:
    """
    Compute HE at standard milestones (24h, 48h, 72h).

    Returns:
        dict with keys 'HE4', 'HE8', 'HE12'
        corresponding to 24h, 48h, 72h
    """
    return {
        'HE4':  heading_error_deg(pred_01deg, gt_01deg, step_idx=3),    # 24h
        'HE8':  heading_error_deg(pred_01deg, gt_01deg, step_idx=7),    # 48h
        'HE12': heading_error_deg(pred_01deg, gt_01deg, step_idx=11),   # 72h
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceResult:
    """Per-sequence evaluation result."""
    ade:       float
    fde:       float
    per_step:  np.ndarray        # [T] km
    dtw:       float
    he4:       float             # Heading Error at 24h (degrees)
    he8:       float             # Heading Error at 48h
    he12:      float             # Heading Error at 72h
    category:  str               # 'straight' | 'recurvature'
    theta:     float             # total rotation angle (degrees)


@dataclass
class DatasetMetrics:
    """
    Aggregated dataset-level metrics — 3-tier framework.

    Tier 1: ADE, FDE (per-step)
    Tier 2: ADE_str, ADE_rec, Δ_rec
    Tier 3: DTW, HE4, HE8, HE12
    """
    # Tier 1
    ade:          float = 0.0
    fde:          float = 0.0
    per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(12))
    per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Horizon breakdown
    h12:  float = 0.0
    h24:  float = 0.0
    h36:  float = 0.0
    h48:  float = 0.0
    h60:  float = 0.0
    h72:  float = 0.0

    # Tier 2
    ade_str:  float = 0.0   # ADE on straight-track sequences
    ade_rec:  float = 0.0   # ADE on recurvature sequences
    delta_rec: float = 0.0  # ADE_rec - ADE_str (straight-track bias)
    n_str:    int   = 0     # count of straight-track sequences
    n_rec:    int   = 0     # count of recurvature sequences

    # Tier 3
    dtw_mean:  float = 0.0
    dtw_str:   float = 0.0
    dtw_rec:   float = 0.0
    he4_mean:  float = 0.0  # mean Heading Error at 24h
    he8_mean:  float = 0.0  # mean Heading Error at 48h
    he12_mean: float = 0.0  # mean Heading Error at 72h
    he4_rec:   float = 0.0  # HE on recurvature only
    he8_rec:   float = 0.0
    he12_rec:  float = 0.0

    n_total: int = 0

    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  TC Track Forecast Metrics",
            "═" * 60,
            f"  Sequences : {self.n_total}"
            f"  (str={self.n_str}, rec={self.n_rec})",
            "",
            "  ── Tier 1: Baseline ─────────────────────────────",
            f"  ADE       : {self.ade:.1f} km",
            f"  FDE       : {self.fde:.1f} km",
            f"  Per-step  : "
            + "  ".join(
                f"{h}h={getattr(self, f'h{h}', 0):.0f}"
                for h in [12, 24, 48, 72]
            ),
            "",
            "  ── Tier 2: Stratified ───────────────────────────",
            f"  ADE_str   : {self.ade_str:.1f} km  (Θ < 45°)",
            f"  ADE_rec   : {self.ade_rec:.1f} km  (Θ ≥ 45°)",
            f"  Δ_rec     : {self.delta_rec:+.1f} km"
            + (" ✅ small" if self.delta_rec < 50 else " ⚠️  large bias"),
            "",
            "  ── Tier 3: Geometric ────────────────────────────",
            f"  DTW (all) : {self.dtw_mean:.1f} km",
            f"  DTW_str   : {self.dtw_str:.1f} km",
            f"  DTW_rec   : {self.dtw_rec:.1f} km",
            f"  HE₄ (24h) : {self.he4_mean:.1f}°"
            f"  [rec={self.he4_rec:.1f}°]",
            f"  HE₈ (48h) : {self.he8_mean:.1f}°"
            f"  [rec={self.he8_rec:.1f}°]",
            f"  HE₁₂(72h) : {self.he12_mean:.1f}°"
            f"  [rec={self.he12_rec:.1f}°]",
            "═" * 60,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class TCEvaluator:
    """
    Full 3-tier TC track evaluator.

    Usage (batch mode during evaluate loop):
        ev = TCEvaluator(pred_len=12)

        for batch in loader:
            pred_norm = model.sample(...)    # [T, B, 2]
            gt_norm   = batch[1]             # [T, B, 2]
            ev.update_batch(pred_norm, gt_norm)

        metrics = ev.compute()
        print(metrics.summary())

    Usage (sequence mode):
        ev = TCEvaluator()
        ev.update_sequence(pred_01deg, gt_01deg)   # [T, 2]
        metrics = ev.compute()
    """

    def __init__(
        self,
        pred_len:    int   = PRED_LEN,
        step_hours:  int   = STEP_HOURS,
        recurv_thr:  float = RECURVATURE_THRESHOLD_DEG,
        compute_dtw: bool  = True,
    ):
        self.pred_len    = pred_len
        self.step_hours  = step_hours
        self.recurv_thr  = recurv_thr
        self.compute_dtw = compute_dtw
        self._results: List[SequenceResult] = []

    def reset(self):
        self._results = []

    def update_sequence(
        self,
        pred_01deg: np.ndarray,
        gt_01deg:   np.ndarray,
    ):
        """
        Add one sequence evaluation result.

        Args:
            pred_01deg: [T, 2+] prediction in 0.1° units
            gt_01deg:   [T, 2+] ground truth in 0.1° units
        """
        ade, fde, per_step = compute_ade_fde(pred_01deg, gt_01deg)

        theta    = total_rotation_angle(gt_01deg)
        category = 'recurvature' if theta >= self.recurv_thr else 'straight'

        dtw_val = (
            dtw_haversine(pred_01deg[:, :2], gt_01deg[:, :2])
            if self.compute_dtw else float('nan')
        )

        he = compute_heading_errors(pred_01deg, gt_01deg)

        self._results.append(SequenceResult(
            ade      = ade,
            fde      = fde,
            per_step = per_step,
            dtw      = dtw_val,
            he4      = he['HE4'],
            he8      = he['HE8'],
            he12     = he['HE12'],
            category = category,
            theta    = theta,
        ))

    def update_batch(
        self,
        pred_norm: torch.Tensor,
        gt_norm:   torch.Tensor,
    ):
        """
        Process a batch of normalized predictions.

        Args:
            pred_norm: [T, B, 2+] normalized (will be denormed internally)
            gt_norm  : [T, B, 2+] normalized
        """
        pred_d = denorm_traj_torch(pred_norm).cpu().numpy()   # [T, B, 2+]
        gt_d   = denorm_traj_torch(gt_norm).cpu().numpy()

        B = pred_d.shape[1]
        for b in range(B):
            self.update_sequence(pred_d[:, b, :], gt_d[:, b, :])

    def compute(self) -> DatasetMetrics:
        """
        Aggregate all sequence results into DatasetMetrics.
        """
        if not self._results:
            return DatasetMetrics()

        results = self._results
        n = len(results)

        # ── Tier 1 ────────────────────────────────────────────────
        all_ade   = np.array([r.ade  for r in results])
        all_fde   = np.array([r.fde  for r in results])
        all_steps = np.stack([r.per_step for r in results], axis=0)  # [N, T]

        per_step_mean = all_steps.mean(axis=0)
        per_step_std  = all_steps.std(axis=0)

        m = DatasetMetrics(
            ade           = float(all_ade.mean()),
            fde           = float(all_fde.mean()),
            per_step_mean = per_step_mean,
            per_step_std  = per_step_std,
            n_total       = n,
        )

        # Per-horizon
        for h, s in HORIZON_STEPS.items():
            if s < self.pred_len:
                setattr(m, f'h{h}', float(per_step_mean[s]))

        # ── Tier 2 ────────────────────────────────────────────────
        str_ades = [r.ade for r in results if r.category == 'straight']
        rec_ades = [r.ade for r in results if r.category == 'recurvature']

        m.n_str   = len(str_ades)
        m.n_rec   = len(rec_ades)
        m.ade_str = float(np.mean(str_ades)) if str_ades else float('nan')
        m.ade_rec = float(np.mean(rec_ades)) if rec_ades else float('nan')
        m.delta_rec = (
            m.ade_rec - m.ade_str
            if not (np.isnan(m.ade_str) or np.isnan(m.ade_rec))
            else float('nan')
        )

        # ── Tier 3 ────────────────────────────────────────────────
        # DTW
        dtw_vals = np.array([r.dtw for r in results if not np.isnan(r.dtw)])
        dtw_str  = np.array([r.dtw for r in results
                             if r.category == 'straight' and not np.isnan(r.dtw)])
        dtw_rec  = np.array([r.dtw for r in results
                             if r.category == 'recurvature' and not np.isnan(r.dtw)])

        m.dtw_mean = float(dtw_vals.mean()) if len(dtw_vals) > 0 else float('nan')
        m.dtw_str  = float(dtw_str.mean())  if len(dtw_str)  > 0 else float('nan')
        m.dtw_rec  = float(dtw_rec.mean())  if len(dtw_rec)  > 0 else float('nan')

        # Heading Errors — all
        def _mean_he(attr):
            vals = [getattr(r, attr) for r in results if not np.isnan(getattr(r, attr))]
            return float(np.mean(vals)) if vals else float('nan')

        def _mean_he_cat(attr, cat):
            vals = [getattr(r, attr) for r in results
                    if r.category == cat and not np.isnan(getattr(r, attr))]
            return float(np.mean(vals)) if vals else float('nan')

        m.he4_mean  = _mean_he('he4')
        m.he8_mean  = _mean_he('he8')
        m.he12_mean = _mean_he('he12')
        m.he4_rec   = _mean_he_cat('he4',  'recurvature')
        m.he8_rec   = _mean_he_cat('he8',  'recurvature')
        m.he12_rec  = _mean_he_cat('he12', 'recurvature')

        return m

    def recurvature_breakdown(self) -> Dict:
        """
        Detailed breakdown of recurvature sequences.
        Useful for debugging which storms are hardest.

        Returns dict with per-sequence stats for recurvature cases.
        """
        rec = [r for r in self._results if r.category == 'recurvature']
        if not rec:
            return {}
        return {
            'count': len(rec),
            'ade':   [r.ade  for r in rec],
            'dtw':   [r.dtw  for r in rec],
            'theta': [r.theta for r in rec],
            'he12':  [r.he12 for r in rec],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Haversine for training loop (fast torch version)
# ══════════════════════════════════════════════════════════════════════════════

class StepErrorAccumulator:
    """
    Efficient weighted accumulator for per-step Haversine errors.
    Use during training/validation loops (fast, no DTW or HE overhead).

    For full evaluation (Tier 2/3), use TCEvaluator.
    """

    def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
        self.pred_len   = pred_len
        self.step_hours = step_hours
        self.reset()

    def reset(self):
        self._sum    = np.zeros(self.pred_len, dtype=np.float64)
        self._sum_sq = np.zeros(self.pred_len, dtype=np.float64)
        self._count  = 0

    def update(self, dist_km: torch.Tensor):
        """
        Args:
            dist_km: [T, B] — Haversine distance per step per sample
        """
        T, B = dist_km.shape
        d = dist_km.double().cpu().numpy()
        self._sum    += d.sum(axis=1)
        self._sum_sq += (d ** 2).sum(axis=1)
        self._count  += B

    def compute(self) -> Dict:
        if self._count == 0:
            return {}

        per_step = self._sum / self._count
        per_step_std = np.sqrt(
            np.maximum(self._sum_sq / self._count - per_step ** 2, 0.0)
        )

        result = {
            'per_step':     per_step,
            'per_step_std': per_step_std,
            'ADE':          float(per_step.mean()),
            'FDE':          float(per_step[-1]),
            'n_samples':    self._count,
        }

        for h, s in HORIZON_STEPS.items():
            if s < self.pred_len:
                result[f'{h}h']     = float(per_step[s])
                result[f'{h}h_std'] = float(per_step_std[s])

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_km — drop-in replacement for training loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_km(
    model,
    loader,
    device,
    num_ensemble: int  = 5,
    ddim_steps:   int  = 10,
    pred_len:     int  = PRED_LEN,
    full_eval:    bool = False,
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate model on a dataloader.

    Args:
        full_eval: if True, run full Tier 2/3 evaluation (slower — includes
                   DTW, HE, stratified metrics). Set False during training
                   for speed.

    Returns:
        metrics  : dict — fast metrics (ADE, FDE, per-horizon)
                   + full metrics if full_eval=True
        per_step : np.ndarray [T]
    """
    import time

    if HAS_TORCH:
        torch.set_grad_enabled(False)
    model.eval()

    # Fast path: Tier 1 only (for training loop)
    acc       = StepErrorAccumulator(pred_len)
    evaluator = TCEvaluator(pred_len=pred_len) if full_eval else None

    t_sample = 0.0
    n_batch  = 0

    for batch in loader:
        bl = _move_batch(list(batch), device)
        gt = bl[1]   # [T, B, 2]

        t0 = time.time()
        pred, _ = model.sample(bl, num_ensemble=num_ensemble, ddim_steps=ddim_steps)
        t_sample += (time.time() - t0) * 1000.0
        n_batch  += 1

        pred_d = denorm_traj_torch(pred)   # [T, B, 2+]
        gt_d   = denorm_traj_torch(gt)

        dist = haversine_km_torch(pred_d, gt_d)   # [T, B]
        acc.update(dist)

        if evaluator is not None:
            evaluator.update_batch(pred, gt)

    fast_metrics = acc.compute()
    fast_metrics['sample_ms_per_batch'] = t_sample / max(n_batch, 1)

    if evaluator is not None:
        dm = evaluator.compute()
        # Merge Tier 2/3 into fast_metrics
        fast_metrics.update({
            'ADE_str':   dm.ade_str,
            'ADE_rec':   dm.ade_rec,
            'delta_rec': dm.delta_rec,
            'n_str':     dm.n_str,
            'n_rec':     dm.n_rec,
            'DTW':       dm.dtw_mean,
            'DTW_str':   dm.dtw_str,
            'DTW_rec':   dm.dtw_rec,
            'HE4':       dm.he4_mean,
            'HE8':       dm.he8_mean,
            'HE12':      dm.he12_mean,
            'HE4_rec':   dm.he4_rec,
            'HE8_rec':   dm.he8_rec,
            'HE12_rec':  dm.he12_rec,
            '_dataset_metrics': dm,
        })

    return fast_metrics, fast_metrics.get('per_step', np.zeros(pred_len))


def _move_batch(bl: list, device) -> list:
    for j, x in enumerate(bl):
        if torch.is_tensor(x):
            bl[j] = x.to(device)
        elif isinstance(x, dict):
            bl[j] = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in x.items()}
    return bl


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy general-purpose metrics (backward compatible)
# ══════════════════════════════════════════════════════════════════════════════

def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.sum((true - pred) ** 2))
                 / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))

def CORR(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(
        ((true - true.mean(0)) ** 2).sum(0)
        * ((pred - pred.mean(0)) ** 2).sum(0)
    ) + 1e-8
    return u / d

def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))

def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))

def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(MSE(pred, true)))

def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-5))))

def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.square((pred - true) / (np.abs(true) + 1e-5))))

def metric(pred: np.ndarray, true: np.ndarray):
    """Tổng hợp cho backward compatibility (intensity metrics)."""
    return (MAE(pred, true), MSE(pred, true), RMSE(pred, true),
            MAPE(pred, true), MSPE(pred, true),
            RSE(pred, true), CORR(pred, true))


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    np.random.seed(42)
    T = 12

    print("=== Self-test: TCEvaluator 3-tier ===\n")

    ev = TCEvaluator(pred_len=T, compute_dtw=True)

    # 18 straight tracks: consistent NW direction
    for i in range(18):
        gt = np.zeros((T, 2))
        gt[:, 0] = np.linspace(1300, 1250, T)    # lon west
        gt[:, 1] = np.linspace(150, 270, T)       # lat north
        pred = gt + np.random.randn(T, 2) * 3.0
        ev.update_sequence(pred, gt)

    # 2 recurvature tracks: sharp turn (WNW → NNE)
    # Use track that makes ~120° turn — theta > 45
    for i in range(2):
        gt = np.zeros((T, 2))
        gt[:, 0] = np.array([1300,1290,1280,1270,1260,1255,
                              1258,1265,1278,1295,1315,1335], dtype=float)
        gt[:, 1] = np.array([150, 160, 175, 192, 210, 228,
                              242, 255, 265, 270, 270, 268], dtype=float)
        pred = gt + np.random.randn(T, 2) * 5.0
        ev.update_sequence(pred, gt)

    m = ev.compute()
    print(m.summary())

    assert m.n_rec == 2,  f"Expected 2 recurvature, got {m.n_rec}"
    assert m.n_str == 18, f"Expected 18 straight, got {m.n_str}"
    assert not np.isnan(m.delta_rec), "Δ_rec is nan"
    print(f"\n  ✅ n_rec={m.n_rec}, n_str={m.n_str}")
    print(f"  ✅ Δ_rec = {m.delta_rec:.1f} km")
    print(f"  ✅ HE₁₂ = {m.he12_mean:.1f}° (mean heading error at 72h)")
    print(f"  ✅ DTW  = {m.dtw_mean:.1f} km")
    print("\nAll tests passed ✅")


if __name__ == '__main__':
    _self_test()