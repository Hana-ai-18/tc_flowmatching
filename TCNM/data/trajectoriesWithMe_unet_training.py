# """
# TCNM/data/trajectoriesWithMe_unet_training.py
# VERSION FIXED - image shape [B, 1, T, 64, 64] cho Unet3D [B, C, D, H, W]

# KEY FIX:
#   seq_collate: img_obs_stack.permute(0, 4, 1, 2, 3) → [B, 1, T_obs, 64, 64]
#   TrajectoryDataset.__getitem__: trả về [n_ped, 2, T] tensors (không gộp toàn bộ)
# """
# import logging
# import os
# import math
# import netCDF4 as nc
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # ══════════════════════════════════════════════════════════
# # SEQ_COLLATE
# # ══════════════════════════════════════════════════════════

# def seq_collate(data):
#     """
#     Input per item:
#       obs_traj   [n_ped, 2, T_obs]   tensor
#       pred_traj  [n_ped, 2, T_pred]  tensor
#       ...
#       img_obs    [T_obs, 64, 64, 1]  tensor   ← channel LAST
#       img_pred   [T_pred, 64, 64, 1] tensor
#       env_data   dict
#       tyID       dict

#     Output slot 11:  [B, 1, T_obs,  64, 64]   ← channel FIRST (Unet3D)
#     Output slot 12:  [B, 1, T_pred, 64, 64]
#     """
#     (obs_traj, pred_traj, obs_rel, pred_rel,
#      non_linear_ped, loss_mask,
#      obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#      obs_date, pred_date,
#      img_obs, img_pred,
#      env_data, tyID) = zip(*data)

#     # ── Trajectory helper ──────────────────────────────────
#     def traj_TBC(lst):
#         """list of [n_ped, 2, T] → cat → [T, total_ped, 2]"""
#         cat = torch.cat(lst, dim=0)       # [total_ped, 2, T]
#         return cat.permute(2, 0, 1)       # [T, total_ped, 2]

#     obs_traj_out    = traj_TBC(obs_traj)
#     pred_traj_out   = traj_TBC(pred_traj)
#     obs_rel_out     = traj_TBC(obs_rel)
#     pred_rel_out    = traj_TBC(pred_rel)
#     obs_Me_out      = traj_TBC(obs_Me)
#     pred_Me_out     = traj_TBC(pred_Me)
#     obs_Me_rel_out  = traj_TBC(obs_Me_rel)
#     pred_Me_rel_out = traj_TBC(pred_Me_rel)

#     # ── non_linear_ped ─────────────────────────────────────
#     nlp_out = torch.tensor(
#         [v for sublist in non_linear_ped for v in
#          (sublist if hasattr(sublist, '__iter__') else [sublist])],
#         dtype=torch.float
#     )

#     # ── loss_mask ──────────────────────────────────────────
#     mask_out = torch.cat(list(loss_mask), dim=0).permute(1, 0)

#     # ── seq_start_end ──────────────────────────────────────
#     counts = torch.tensor([t.shape[0] for t in obs_traj])
#     cum    = torch.cumsum(counts, dim=0)
#     starts = torch.cat([torch.tensor([0]), cum[:-1]])
#     seq_start_end = torch.stack([starts, cum], dim=1)

#     # ── dates ─────────────────────────────────────────────
#     obs_date_out  = obs_date[0]
#     pred_date_out = pred_date[0]

#     # ── Images ────────────────────────────────────────────
#     # img_obs: each item is [T_obs, 64, 64, 1]
#     # Stack → [B, T_obs, 64, 64, 1] → permute → [B, 1, T_obs, 64, 64]
#     img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
#     img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()
#     # Shape check: should be [B, 1, T, 64, 64]
#     assert img_obs_out.shape[1] == 1, \
#         f"Image channel dim wrong: {img_obs_out.shape}. Expected [B,1,T,64,64]"

#     # ── env_data ──────────────────────────────────────────
#     if env_data[0] is None:
#         env_out = None
#     else:
#         env_out = {}
#         all_keys = set()
#         for d in env_data:
#             if isinstance(d, dict):
#                 all_keys.update(d.keys())
#         for key in all_keys:
#             if key == 'location':
#                 continue
#             vals = []
#             for d in env_data:
#                 if not isinstance(d, dict) or key not in d:
#                     continue
#                 v = d[key]
#                 if not torch.is_tensor(v):
#                     v = torch.tensor(v, dtype=torch.float)
#                 vals.append(v.float())
#             if not vals:
#                 continue
#             try:
#                 env_out[key] = torch.stack(vals, dim=0)
#             except RuntimeError:
#                 mx = max(v.numel() for v in vals)
#                 padded = [torch.nn.functional.pad(v.flatten(), (0, mx - v.numel()))
#                           for v in vals]
#                 env_out[key] = torch.stack(padded, dim=0)

#     return (
#         obs_traj_out,       # 0  [T_obs,  B, 2]
#         pred_traj_out,      # 1  [T_pred, B, 2]
#         obs_rel_out,        # 2
#         pred_rel_out,       # 3
#         nlp_out,            # 4  [total_ped]
#         mask_out,           # 5  [T_seq, total_ped]
#         seq_start_end,      # 6  [B, 2]
#         obs_Me_out,         # 7  [T_obs,  B, 2]
#         pred_Me_out,        # 8  [T_pred, B, 2]
#         obs_Me_rel_out,     # 9
#         pred_Me_rel_out,    # 10
#         img_obs_out,        # 11 [B, 1, T_obs,  64, 64]  ← FIXED
#         img_pred_out,       # 12 [B, 1, T_pred, 64, 64]
#         env_out,            # 13 dict or None
#         None,               # 14 placeholder
#         list(tyID),         # 15
#     )


# # ══════════════════════════════════════════════════════════
# # DATASET
# # ══════════════════════════════════════════════════════════

# class TrajectoryDataset(Dataset):
#     """
#     TC trajectory dataset.
#     File format: WP{YEAR}{TYPHOON}_{TIMESTAMP}.npy / .nc
#     """

#     def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
#                  threshold=0.002, min_ped=1, delim=' ',
#                  other_modal='gph', test_year=None,
#                  type='train', is_test=False, **kwargs):
#         super().__init__()

#         # ── Path resolution ────────────────────────────────
#         if isinstance(data_dir, dict):
#             root  = data_dir['root']
#             dtype = data_dir.get('type', type)
#         else:
#             root  = data_dir
#             dtype = type
#         if is_test:
#             dtype = 'test'

#         root = os.path.abspath(root)
#         bn   = os.path.basename(root)

#         if bn in ('train', 'test'):
#             # root = .../Data1d/train  → go up twice
#             self.root_path = os.path.dirname(os.path.dirname(root))
#         elif bn == 'Data1d':
#             self.root_path = os.path.dirname(root)
#         else:
#             self.root_path = root

#         self.data1d_path = os.path.join(self.root_path, 'Data1d', dtype)
#         self.data3d_path = os.path.join(self.root_path, 'Data3d')
#         self.env_path    = os.path.join(self.root_path, 'ENV_DATA')

#         logger.info(f"📂 Using root as-is: {self.root_path}")
#         logger.info(f"   Data1d ({dtype}): {self.data1d_path}")
#         logger.info(f"   Data3d: {self.data3d_path}")
#         logger.info(f"   ENV: {self.env_path}")

#         self.obs_len    = obs_len
#         self.pred_len   = pred_len
#         self.skip       = skip
#         self.seq_len    = obs_len + pred_len
#         self.modal_name = other_modal
#         self.test_year  = test_year

#         if not os.path.exists(self.data1d_path):
#             logger.error(f"❌ Missing: {self.data1d_path}")
#             self.num_seq = 0
#             self.seq_start_end = []
#             return

#         all_files = [
#             os.path.join(self.data1d_path, f)
#             for f in os.listdir(self.data1d_path)
#             if f.endswith('.txt')
#             and (test_year is None or str(test_year) in f)
#         ]
#         logger.info(f"✅ Found {len(all_files)} matching files")
#         if test_year:
#             logger.info(f"   Filtered by year: {test_year}")
#         if all_files:
#             logger.info("   Sample files:")
#             for f in all_files[:3]:
#                 logger.info(f"      - {os.path.basename(f)}")

#         # Per-ped storage (avoids reshape issues in __getitem__)
#         self.obs_traj_raw   = []  # each: [2, obs_len]
#         self.pred_traj_raw  = []
#         self.obs_Me_raw     = []
#         self.pred_Me_raw    = []
#         self.obs_rel_raw    = []
#         self.pred_rel_raw   = []
#         self.non_linear_ped = []
#         self.tyID           = []
#         num_peds_in_seq     = []

#         for path in all_files:
#             base   = os.path.splitext(os.path.basename(path))[0]
#             parts  = base.split('_')
#             f_year = parts[0] if len(parts) > 0 else 'unknown'
#             f_name = parts[1] if len(parts) > 1 else base

#             d    = self._read_file(path, delim)
#             data = d['main']
#             add  = d['addition']
#             if len(data) < self.seq_len:
#                 continue

#             frames     = np.unique(data[:, 0]).tolist()
#             frame_data = [data[data[:, 0] == f] for f in frames]
#             n_seq      = int(math.ceil(
#                 (len(frames) - self.seq_len + 1) / self.skip))

#             for idx in range(0, n_seq * self.skip, self.skip):
#                 if idx + self.seq_len > len(frame_data):
#                     break
#                 seg  = np.concatenate(frame_data[idx:idx + self.seq_len])
#                 peds = np.unique(seg[:, 1])
#                 cnt  = 0

#                 for pid in peds:
#                     ps = seg[seg[:, 1] == pid]
#                     if len(ps) != self.seq_len:
#                         continue
#                     ps  = np.transpose(ps[:, 2:])   # [4, seq_len]
#                     rel = np.zeros_like(ps)
#                     rel[:, 1:] = ps[:, 1:] - ps[:, :-1]

#                     self.obs_traj_raw.append(
#                         torch.from_numpy(ps[:2, :obs_len]).float())
#                     self.pred_traj_raw.append(
#                         torch.from_numpy(ps[:2, obs_len:]).float())
#                     self.obs_Me_raw.append(
#                         torch.from_numpy(ps[2:, :obs_len]).float())
#                     self.pred_Me_raw.append(
#                         torch.from_numpy(ps[2:, obs_len:]).float())
#                     self.obs_rel_raw.append(
#                         torch.from_numpy(rel[:2, :obs_len]).float())
#                     self.pred_rel_raw.append(
#                         torch.from_numpy(rel[:2, obs_len:]).float())
#                     self.non_linear_ped.append(
#                         self._poly_fit(ps, pred_len, threshold))
#                     cnt += 1

#                 if cnt >= min_ped:
#                     num_peds_in_seq.append(cnt)
#                     self.tyID.append({
#                         'old':    [f_year, f_name, idx],
#                         'tydate': [add[i][0]
#                                    for i in range(idx, idx + self.seq_len)]
#                     })

#         self.num_seq = len(self.tyID)
#         cum = np.cumsum(num_peds_in_seq).tolist()
#         self.seq_start_end = list(zip([0] + cum[:-1], cum))
#         logger.info(f"✅ Successfully loaded {self.num_seq} sequences")

#     # ── Helpers ───────────────────────────────────────────

#     def _read_file(self, path, delim):
#         data, add = [], []
#         with open(path) as f:
#             for line in f:
#                 p = line.strip().split(delim)
#                 if len(p) < 5:
#                     continue
#                 add.append(p[-2:])
#                 nums = [
#                     1.0 if i == 1
#                     else (float(v) if v.lower() != 'null' else 0.0)
#                     for i, v in enumerate(p[:-2])
#                 ]
#                 data.append(nums)
#         return {'main': np.asarray(data), 'addition': add}

#     def _poly_fit(self, traj, tlen, threshold):
#         t  = np.linspace(0, tlen - 1, tlen)
#         rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
#         ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
#         return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

#     def _load_nc(self, path):
#         try:
#             with nc.Dataset(path) as ds:
#                 key = list(ds.variables.keys())[-1]
#                 arr = np.array(ds.variables[key][:])
#             if arr.ndim == 3:
#                 arr = arr[0]
#             arr = cv2.resize(arr.astype(np.float32), (64, 64))
#             arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
#             return torch.from_numpy(arr).float().unsqueeze(-1)
#         except Exception:
#             return None

#     def _load_npy(self, path):
#         try:
#             arr = np.load(path)
#             if arr.ndim == 3:
#                 arr = arr[0]
#             arr = cv2.resize(arr.astype(np.float32), (64, 64))
#             arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
#             return torch.from_numpy(arr).float().unsqueeze(-1)
#         except Exception:
#             return None

#     def img_read(self, year, ty_name, timestamp):
#         """Load satellite image → [64, 64, 1] float tensor."""
#         folder = os.path.join(self.data3d_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return torch.zeros(64, 64, 1)

#         prefix = f"WP{year}{ty_name}_{timestamp}"
#         for ext, fn in [('.nc', self._load_nc), ('.npy', self._load_npy)]:
#             p = os.path.join(folder, prefix + ext)
#             if os.path.exists(p):
#                 r = fn(p)
#                 if r is not None:
#                     return r

#         # timestamp fallback
#         try:
#             for f in sorted(os.listdir(folder)):
#                 if timestamp in f:
#                     p = os.path.join(folder, f)
#                     fn = self._load_nc if f.endswith('.nc') else self._load_npy
#                     r  = fn(p)
#                     if r is not None:
#                         return r
#         except Exception:
#             pass
#         return torch.zeros(64, 64, 1)

#     def _get_env(self, year, ty_name, timestamp):
#         folder = os.path.join(self.env_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return {'wind': 0.0, 'month': 0.0, 'move_velocity': 0.0}

#         fname = f"WP{year}{ty_name}_{timestamp}.npy"
#         path  = os.path.join(folder, fname)
#         if not os.path.exists(path):
#             cands = [f for f in os.listdir(folder)
#                      if timestamp in f and f.endswith('.npy')]
#             path  = os.path.join(folder, cands[0]) if cands else None

#         if path and os.path.exists(path):
#             try:
#                 d = np.load(path, allow_pickle=True).item()
#                 return {k: (0.0 if v == -1 else v) for k, v in d.items()}
#             except Exception:
#                 pass
#         return {'wind': 0.0, 'month': 0.0, 'move_velocity': 0.0}

#     def _embed_time(self, date_list):
#         rows = []
#         for d in date_list:
#             try:
#                 rows.append([
#                     (float(d[:4]) - 1949) / 70.0 - 0.5,
#                     (float(d[4:6]) - 1) / 11.0 - 0.5,
#                     (float(d[6:8]) - 1) / 30.0 - 0.5,
#                     float(d[8:10]) / 18.0 - 0.5,
#                 ])
#             except Exception:
#                 rows.append([0., 0., 0., 0.])
#         arr = torch.tensor(rows, dtype=torch.float).t()  # [4, T]
#         return arr.unsqueeze(0)                          # [1, 4, T]

#     # ── __len__ / __getitem__ ────────────────────────────

#     def __len__(self):
#         return self.num_seq

#     def __getitem__(self, index):
#         if self.num_seq == 0:
#             raise IndexError("Empty dataset")

#         s, e   = self.seq_start_end[index]
#         info   = self.tyID[index]
#         year   = str(info['old'][0])
#         tyname = str(info['old'][1])
#         dates  = info['tydate']

#         # Images: list of [64, 64, 1] → stack → [T_obs, 64, 64, 1]
#         imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
#         img_obs = torch.stack(imgs, dim=0)                        # [T_obs, 64, 64, 1]
#         img_pred = torch.zeros(self.pred_len, 64, 64, 1)

#         # Env data
#         envs = [self._get_env(year, tyname, ts) for ts in dates[:self.obs_len]]
#         env_out = {}
#         if envs:
#             for k in envs[0]:
#                 if k == 'location':
#                     continue
#                 vs = []
#                 for d in envs:
#                     v = d.get(k, 0.0)
#                     if isinstance(v, (int, float)):
#                         v = torch.tensor([float(v)])
#                     elif isinstance(v, np.ndarray):
#                         v = torch.from_numpy(v.astype(np.float32)).reshape(-1)
#                     elif not torch.is_tensor(v):
#                         v = torch.tensor(v, dtype=torch.float).reshape(-1)
#                     vs.append(v)
#                 try:
#                     env_out[k] = torch.stack(vs, dim=0)   # [T_obs, feat]
#                 except Exception:
#                     pass

#         # Per-ped tensors: each [2, T]
#         obs_traj   = torch.stack([self.obs_traj_raw[i]  for i in range(s, e)])
#         pred_traj  = torch.stack([self.pred_traj_raw[i] for i in range(s, e)])
#         obs_rel    = torch.stack([self.obs_rel_raw[i]   for i in range(s, e)])
#         pred_rel   = torch.stack([self.pred_rel_raw[i]  for i in range(s, e)])
#         obs_Me     = torch.stack([self.obs_Me_raw[i]    for i in range(s, e)])
#         pred_Me    = torch.stack([self.pred_Me_raw[i]   for i in range(s, e)])
#         n          = e - s
#         nlp        = [self.non_linear_ped[i] for i in range(s, e)]
#         mask       = torch.ones(n, self.seq_len)

#         return [
#             obs_traj,   pred_traj,  obs_rel,   pred_rel,
#             nlp,        mask,
#             obs_Me,     pred_Me,    obs_rel,   pred_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs,    img_pred,
#             env_out,    info,
#         ]

"""
TCNM/data/trajectoriesWithMe_unet_training.py
VERSION FIXED — image shape [B, 1, T, 64, 64] cho Unet3D [B, C, D, H, W]

KEY FIXES:
  seq_collate: img_obs_stack.permute(0, 4, 1, 2, 3) → [B, 1, T_obs, 64, 64]
  TrajectoryDataset.__getitem__: trả về [n_ped, 2, T] tensors
  env_data_processing: exported để trajectoriesWithMe_unet.py có thể import
"""
import logging
import os
import math
import netCDF4 as nc
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# ENV DATA PROCESSING (exported — used by inference dataset)
# ══════════════════════════════════════════════════════════

def env_data_processing(env_dict: dict) -> dict:
    """
    Normalise / clean env dict loaded from .npy files.
    Replace sentinel -1 values with 0.0.
    """
    if not isinstance(env_dict, dict):
        return {}
    return {k: (0.0 if v == -1 else v) for k, v in env_dict.items()}


# ══════════════════════════════════════════════════════════
# SEQ_COLLATE
# ══════════════════════════════════════════════════════════

def seq_collate(data):
    """
    Input per item (16 elements):
      [0]  obs_traj   [n_ped, 2, T_obs]   tensor
      [1]  pred_traj  [n_ped, 2, T_pred]  tensor
      [2]  obs_rel    [n_ped, 2, T_obs]
      [3]  pred_rel   [n_ped, 2, T_pred]
      [4]  non_linear_ped  scalar or list
      [5]  loss_mask  [n_ped, seq_len]
      [6]  obs_Me     [n_ped, 2, T_obs]
      [7]  pred_Me    [n_ped, 2, T_pred]
      [8]  obs_Me_rel [n_ped, 2, T_obs]
      [9]  pred_Me_rel[n_ped, 2, T_pred]
      [10] obs_date   tensor
      [11] pred_date  tensor
      [12] img_obs    [T_obs, 64, 64, 1]  ← channel LAST
      [13] img_pred   [T_pred, 64, 64, 1]
      [14] env_data   dict or None
      [15] tyID       dict

    Output slot 11:  [B, 1, T_obs,  64, 64]  ← channel FIRST (UNet3D)
    Output slot 12:  [B, 1, T_pred, 64, 64]
    """
    (obs_traj, pred_traj, obs_rel, pred_rel,
     non_linear_ped, loss_mask,
     obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
     obs_date, pred_date,
     img_obs, img_pred,
     env_data, tyID) = zip(*data)

    # ── Trajectory helper ──────────────────────────────────
    def traj_TBC(lst):
        """list of [n_ped, 2, T] → cat → [T, total_ped, 2]"""
        cat = torch.cat(lst, dim=0)   # [total_ped, 2, T]
        return cat.permute(2, 0, 1)   # [T, total_ped, 2]

    obs_traj_out    = traj_TBC(obs_traj)
    pred_traj_out   = traj_TBC(pred_traj)
    obs_rel_out     = traj_TBC(obs_rel)
    pred_rel_out    = traj_TBC(pred_rel)
    obs_Me_out      = traj_TBC(obs_Me)
    pred_Me_out     = traj_TBC(pred_Me)
    obs_Me_rel_out  = traj_TBC(obs_Me_rel)
    pred_Me_rel_out = traj_TBC(pred_Me_rel)

    # ── non_linear_ped ─────────────────────────────────────
    nlp_out = torch.tensor(
        [v for sublist in non_linear_ped for v in
         (sublist if hasattr(sublist, '__iter__') else [sublist])],
        dtype=torch.float
    )

    # ── loss_mask ──────────────────────────────────────────
    mask_out = torch.cat(list(loss_mask), dim=0).permute(1, 0)

    # ── seq_start_end ──────────────────────────────────────
    counts = torch.tensor([t.shape[0] for t in obs_traj])
    cum    = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0]), cum[:-1]])
    seq_start_end = torch.stack([starts, cum], dim=1)

    # ── dates ─────────────────────────────────────────────
    obs_date_out  = obs_date[0]
    pred_date_out = pred_date[0]

    # ── Images ───────────────────────────────────────────
    # img_obs: each item is [T_obs, 64, 64, 1]
    # Stack → [B, T_obs, 64, 64, 1] → permute → [B, 1, T_obs, 64, 64]
    img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()
    assert img_obs_out.shape[1] == 1, \
        f"Image channel dim wrong: {img_obs_out.shape}. Expected [B,1,T,64,64]"

    # ── env_data ──────────────────────────────────────────
    if env_data[0] is None:
        env_out = None
    else:
        env_out  = {}
        all_keys = set()
        for d in env_data:
            if isinstance(d, dict):
                all_keys.update(d.keys())
        for key in all_keys:
            if key == 'location':
                continue
            vals = []
            for d in env_data:
                if not isinstance(d, dict) or key not in d:
                    continue
                v = d[key]
                if not torch.is_tensor(v):
                    v = torch.tensor(v, dtype=torch.float)
                vals.append(v.float())
            if not vals:
                continue
            try:
                env_out[key] = torch.stack(vals, dim=0)
            except RuntimeError:
                mx     = max(v.numel() for v in vals)
                padded = [torch.nn.functional.pad(v.flatten(), (0, mx - v.numel()))
                          for v in vals]
                env_out[key] = torch.stack(padded, dim=0)

    return (
        obs_traj_out,       # 0  [T_obs,  B, 2]
        pred_traj_out,      # 1  [T_pred, B, 2]
        obs_rel_out,        # 2
        pred_rel_out,       # 3
        nlp_out,            # 4  [total_ped]
        mask_out,           # 5  [T_seq, total_ped]
        seq_start_end,      # 6  [B, 2]
        obs_Me_out,         # 7  [T_obs,  B, 2]
        pred_Me_out,        # 8  [T_pred, B, 2]
        obs_Me_rel_out,     # 9
        pred_Me_rel_out,    # 10
        img_obs_out,        # 11 [B, 1, T_obs,  64, 64]  ← FIXED
        img_pred_out,       # 12 [B, 1, T_pred, 64, 64]
        env_out,            # 13 dict or None
        None,               # 14 placeholder
        list(tyID),         # 15
    )


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """
    TC trajectory dataset for training.
    File format: YEAR_TYPHOONNAME.txt
    Each line: STT  1.0  LONG_norm  LAT_norm  PRES_norm  WIND_norm  DATE  NAME
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 threshold=0.002, min_ped=1, delim=' ',
                 other_modal='gph', test_year=None,
                 type='train', is_test=False, **kwargs):
        super().__init__()

        # ── Path resolution ────────────────────────────────
        if isinstance(data_dir, dict):
            root  = data_dir['root']
            dtype = data_dir.get('type', type)
        else:
            root  = data_dir
            dtype = type
        if is_test:
            dtype = 'test'

        root = os.path.abspath(root)
        bn   = os.path.basename(root)

        if bn in ('train', 'test', 'val'):
            self.root_path = os.path.dirname(os.path.dirname(root))
        elif bn == 'Data1d':
            self.root_path = os.path.dirname(root)
        else:
            self.root_path = root

        self.data1d_path = os.path.join(self.root_path, 'Data1d', dtype)
        self.data3d_path = os.path.join(self.root_path, 'Data3d')
        self.env_path    = os.path.join(self.root_path, 'ENV_DATA')

        logger.info(f"📂 root: {self.root_path}")
        logger.info(f"   Data1d ({dtype}): {self.data1d_path}")
        logger.info(f"   Data3d: {self.data3d_path}")
        logger.info(f"   ENV: {self.env_path}")

        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.skip       = skip
        self.seq_len    = obs_len + pred_len
        self.modal_name = other_modal
        self.test_year  = test_year

        if not os.path.exists(self.data1d_path):
            logger.error(f"❌ Missing: {self.data1d_path}")
            self.num_seq = 0
            self.seq_start_end = []
            return

        all_files = [
            os.path.join(self.data1d_path, f)
            for f in os.listdir(self.data1d_path)
            if f.endswith('.txt')
            and (test_year is None or str(test_year) in f)
        ]
        logger.info(f"✅ {len(all_files)} files (year filter: {test_year})")

        # Per-ped storage
        self.obs_traj_raw   = []
        self.pred_traj_raw  = []
        self.obs_Me_raw     = []
        self.pred_Me_raw    = []
        self.obs_rel_raw    = []
        self.pred_rel_raw   = []
        self.non_linear_ped = []
        self.tyID           = []
        num_peds_in_seq     = []

        for path in all_files:
            base   = os.path.splitext(os.path.basename(path))[0]
            parts  = base.split('_')
            f_year = parts[0] if len(parts) > 0 else 'unknown'
            f_name = parts[1] if len(parts) > 1 else base

            d    = self._read_file(path, delim)
            data = d['main']
            add  = d['addition']
            if len(data) < self.seq_len:
                continue

            frames     = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == f] for f in frames]
            n_seq      = int(math.ceil(
                (len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, n_seq * self.skip, self.skip):
                if idx + self.seq_len > len(frame_data):
                    break
                seg  = np.concatenate(frame_data[idx:idx + self.seq_len])
                peds = np.unique(seg[:, 1])
                cnt  = 0

                for pid in peds:
                    ps = seg[seg[:, 1] == pid]
                    if len(ps) != self.seq_len:
                        continue
                    ps  = np.transpose(ps[:, 2:])   # [4, seq_len]
                    rel = np.zeros_like(ps)
                    rel[:, 1:] = ps[:, 1:] - ps[:, :-1]

                    self.obs_traj_raw.append(
                        torch.from_numpy(ps[:2, :obs_len]).float())
                    self.pred_traj_raw.append(
                        torch.from_numpy(ps[:2, obs_len:]).float())
                    self.obs_Me_raw.append(
                        torch.from_numpy(ps[2:, :obs_len]).float())
                    self.pred_Me_raw.append(
                        torch.from_numpy(ps[2:, obs_len:]).float())
                    self.obs_rel_raw.append(
                        torch.from_numpy(rel[:2, :obs_len]).float())
                    self.pred_rel_raw.append(
                        torch.from_numpy(rel[:2, obs_len:]).float())
                    self.non_linear_ped.append(
                        self._poly_fit(ps, pred_len, threshold))
                    cnt += 1

                if cnt >= min_ped:
                    num_peds_in_seq.append(cnt)
                    self.tyID.append({
                        'old':    [f_year, f_name, idx],
                        'tydate': [add[i][0]
                                   for i in range(idx, idx + self.seq_len)]
                    })

        self.num_seq = len(self.tyID)
        cum = np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = list(zip([0] + cum[:-1], cum))
        logger.info(f"✅ {self.num_seq} sequences loaded")

    # ── Helpers ───────────────────────────────────────────

    def _read_file(self, path, delim):
        data, add = [], []
        with open(path) as f:
            for line in f:
                p = line.strip().split(delim)
                if len(p) < 5:
                    continue
                add.append(p[-2:])
                nums = [
                    1.0 if i == 1
                    else (float(v) if v.lower() != 'null' else 0.0)
                    for i, v in enumerate(p[:-2])
                ]
                data.append(nums)
        return {'main': np.asarray(data), 'addition': add}

    def _poly_fit(self, traj, tlen, threshold):
        t  = np.linspace(0, tlen - 1, tlen)
        rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
        ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
        return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

    def _load_nc(self, path):
        try:
            with nc.Dataset(path) as ds:
                key = list(ds.variables.keys())[-1]
                arr = np.array(ds.variables[key][:])
            if arr.ndim == 3:
                arr = arr[0]
            arr = cv2.resize(arr.astype(np.float32), (64, 64))
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            return torch.from_numpy(arr).float().unsqueeze(-1)
        except Exception:
            return None

    def _load_npy(self, path):
        try:
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr[0]
            arr = cv2.resize(arr.astype(np.float32), (64, 64))
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            return torch.from_numpy(arr).float().unsqueeze(-1)
        except Exception:
            return None

    def img_read(self, year, ty_name, timestamp):
        """Load satellite image → [64, 64, 1] float tensor."""
        folder = os.path.join(self.data3d_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return torch.zeros(64, 64, 1)

        prefix = f"WP{year}{ty_name}_{timestamp}"
        for ext, fn in [('.nc', self._load_nc), ('.npy', self._load_npy)]:
            p = os.path.join(folder, prefix + ext)
            if os.path.exists(p):
                r = fn(p)
                if r is not None:
                    return r

        # Timestamp fallback
        try:
            for fname in sorted(os.listdir(folder)):
                if timestamp in fname:
                    p  = os.path.join(folder, fname)
                    fn = self._load_nc if fname.endswith('.nc') else self._load_npy
                    r  = fn(p)
                    if r is not None:
                        return r
        except Exception:
            pass
        return torch.zeros(64, 64, 1)

    def _get_env(self, year, ty_name, timestamp):
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return {'wind': 0.0, 'month': 0.0, 'move_velocity': 0.0}

        fname = f"WP{year}{ty_name}_{timestamp}.npy"
        path  = os.path.join(folder, fname)
        if not os.path.exists(path):
            cands = [f for f in os.listdir(folder)
                     if timestamp in f and f.endswith('.npy')]
            path  = os.path.join(folder, cands[0]) if cands else None

        if path and os.path.exists(path):
            try:
                d = np.load(path, allow_pickle=True).item()
                return env_data_processing(d)
            except Exception:
                pass
        return {'wind': 0.0, 'month': 0.0, 'move_velocity': 0.0}

    def _embed_time(self, date_list):
        rows = []
        for d in date_list:
            try:
                rows.append([
                    (float(d[:4]) - 1949) / 70.0 - 0.5,
                    (float(d[4:6]) - 1)  / 11.0 - 0.5,
                    (float(d[6:8]) - 1)  / 30.0 - 0.5,
                    float(d[8:10])       / 18.0 - 0.5,
                ])
            except Exception:
                rows.append([0., 0., 0., 0.])
        arr = torch.tensor(rows, dtype=torch.float).t()   # [4, T]
        return arr.unsqueeze(0)                            # [1, 4, T]

    # ── __len__ / __getitem__ ────────────────────────────

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        if self.num_seq == 0:
            raise IndexError("Empty dataset")

        s, e   = self.seq_start_end[index]
        info   = self.tyID[index]
        year   = str(info['old'][0])
        tyname = str(info['old'][1])
        dates  = info['tydate']

        # Images: [T_obs, 64, 64, 1]
        imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
        img_obs = torch.stack(imgs, dim=0)
        img_pred = torch.zeros(self.pred_len, 64, 64, 1)

        # Env data
        envs    = [self._get_env(year, tyname, ts) for ts in dates[:self.obs_len]]
        env_out = {}
        if envs:
            for k in envs[0]:
                if k == 'location':
                    continue
                vs = []
                for d in envs:
                    v = d.get(k, 0.0)
                    if isinstance(v, (int, float)):
                        v = torch.tensor([float(v)])
                    elif isinstance(v, np.ndarray):
                        v = torch.from_numpy(v.astype(np.float32)).reshape(-1)
                    elif not torch.is_tensor(v):
                        v = torch.tensor(v, dtype=torch.float).reshape(-1)
                    vs.append(v)
                try:
                    env_out[k] = torch.stack(vs, dim=0)
                except Exception:
                    pass

        # Per-ped tensors [n_ped, 2, T]
        obs_traj  = torch.stack([self.obs_traj_raw[i]  for i in range(s, e)])
        pred_traj = torch.stack([self.pred_traj_raw[i] for i in range(s, e)])
        obs_rel   = torch.stack([self.obs_rel_raw[i]   for i in range(s, e)])
        pred_rel  = torch.stack([self.pred_rel_raw[i]  for i in range(s, e)])
        obs_Me    = torch.stack([self.obs_Me_raw[i]    for i in range(s, e)])
        pred_Me   = torch.stack([self.pred_Me_raw[i]   for i in range(s, e)])
        n         = e - s
        nlp       = [self.non_linear_ped[i] for i in range(s, e)]
        mask      = torch.ones(n, self.seq_len)

        return [
            obs_traj,  pred_traj, obs_rel,  pred_rel,
            nlp,       mask,
            obs_Me,    pred_Me,   obs_rel,  pred_rel,
            self._embed_time(dates[:self.obs_len]),
            self._embed_time(dates[self.obs_len:]),
            img_obs,   img_pred,
            env_out,   info,
        ]