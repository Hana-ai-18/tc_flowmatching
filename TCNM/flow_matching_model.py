# """
# TCNM/flow_matching_model.py  ── v5  OT-CFM + PINN + Absolute Trajectory
# =========================================================================

# ROOT CAUSE của straight-line prediction:
#   v4 dùng cumulative displacement (cumsum) → model chỉ cần học
#   "mỗi bước đi bao xa theo hướng hiện tại" → extrapolate thẳng là tối ưu.
#   Với recurvature, bão đổi hướng đột ngột — displacement thay đổi hoàn toàn
#   nhưng model không có signal để dự đoán điều đó.

# FIXES vs v4:
# ──────────────────────────────────────────────────────────────────────────
# FIX 1 │ traj_to_rel: cumulative displacement → absolute offset từ last_pos
#        │ - v4: x1[t] = Σ displacement[0..t]  → mỗi step tích lũy sai số
#        │ - v5: x1[t] = pos[t] - last_pos     → model học hình dạng toàn
#        │   bộ trajectory, recurvature là 1 pattern shape có thể học được
#        │ - rel_to_abs: bỏ cumsum, chỉ cộng offset vào last_pos
# ──────────────────────────────────────────────────────────────────────────
# FIX 2 │ _dir_loss: thêm step-wise direction loss
#        │ - v4: chỉ so hướng tổng thể pred vs gt từ last_pos
#        │   → không phạt khi đúng hướng đầu nhưng sai hướng sau recurvature
#        │ - v5: so cosine(pred_velocity[t], gt_velocity[t]) từng bước
#        │   → phạt nặng khi hướng di chuyển từng bước sai > 60°
# ──────────────────────────────────────────────────────────────────────────
# FIX 3 │ _smooth_loss: tính trên velocity/acceleration của abs trajectory
#        │ - v4: tính trên rel tensor [B,T,4] → đo smoothness của displacement
#        │ - v5: tính acceleration = Δvelocity → phạt đổi hướng ĐỘT NGỘT
#        │   (không phạt recurvature dần dần, chỉ phạt noise/jitter)
# ──────────────────────────────────────────────────────────────────────────

# Giữ nguyên: VelocityField, OT-CFM training, PINN vorticity, sampling
# """
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from TCNM.Unet3D_merge_tiny import Unet3D
# from TCNM.env_net_transformer_gphsplit import Env_net


# # ── Physical constants ────────────────────────────────────────────────────────
# OMEGA      = 7.2921e-5
# R_EARTH    = 6.371e6
# DT_6H      = 6 * 3600
# NORM_TO_MS = 555e3 / DT_6H


# # ── Velocity Field Network ────────────────────────────────────────────────────
# class VelocityField(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=128, sigma_min=0.001):
#         super().__init__()
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.sigma_min = sigma_min

#         self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
#         self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
#         self.obs_lstm     = nn.LSTM(input_size=4, hidden_size=128,
#                                     num_layers=3, batch_first=True, dropout=0.2)
#         self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
#         self.ctx_fc1      = nn.Linear(16 + 64 + 128, 512)
#         self.ctx_ln       = nn.LayerNorm(512)
#         self.ctx_drop     = nn.Dropout(0.15)
#         self.ctx_fc2      = nn.Linear(512, ctx_dim)

#         self.time_fc1   = nn.Linear(128, 256)
#         self.time_fc2   = nn.Linear(256, 128)
#         self.traj_embed = nn.Linear(4, 128)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=128, nhead=8, dim_feedforward=512,
#             dropout=0.15, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
#         self.out_fc1 = nn.Linear(128, 256)
#         self.out_fc2 = nn.Linear(256, 4)

#     def _time_emb(self, t, dim=128):
#         device = t.device
#         half   = dim // 2
#         freq   = torch.exp(
#             torch.arange(half, dtype=torch.float32, device=device)
#             * (-math.log(10000.0) / (half - 1))
#         )
#         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return F.pad(emb, (0, 1)) if dim % 2 else emb

#     def _extract_context(self, batch_list):
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]

#         f_s = self.spatial_enc(image_obs).mean(dim=2)
#         f_s = self.spatial_pool(f_s).flatten(1)
#         f_e, _, _ = self.env_enc(env_data, image_obs)

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         _, (h_n, _) = self.obs_lstm(obs_in)
#         f_h = h_n[-1]

#         ctx = torch.cat([f_s, f_e, f_h], dim=-1)
#         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
#         ctx = self.ctx_drop(ctx)
#         ctx = self.ctx_fc2(ctx)
#         return ctx

#     def forward(self, x_t, t, batch_list):
#         ctx   = self._extract_context(batch_list)
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)
#         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
#         out    = self.transformer(x_emb, memory)
#         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # ── TCFlowMatching ────────────────────────────────────────────────────────────
# class TCFlowMatching(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, num_steps=100,
#                  sigma_min=0.001, **kwargs):
#         super().__init__()
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.sigma_min = sigma_min
#         self.net       = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

#     # ── FIX 1: Absolute offset thay vì cumulative displacement ────────────────
#     @staticmethod
#     def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
#         """
#         v5: encode trajectory as absolute offset from last observed position.
#         Model học hình dạng toàn bộ trajectory → học được recurvature.

#         v4 cũ: cumulative displacement → model extrapolate thẳng
#         v5 mới: offset từ last_pos → model học absolute shape
#         """
#         traj_norm = traj_gt - last_pos.unsqueeze(0)   # [T, B, 2]
#         me_norm   = Me_gt   - last_Me.unsqueeze(0)    # [T, B, 2]
#         return torch.cat([traj_norm, me_norm], dim=-1).permute(1, 0, 2)  # [B, T, 4]

#     @staticmethod
#     def rel_to_abs(rel, last_pos, last_Me):
#         """Inverse of traj_to_rel: offset + last_pos → absolute position."""
#         d    = rel.permute(1, 0, 2)            # [T, B, 4]
#         traj = last_pos.unsqueeze(0) + d[:, :, :2]   # không cumsum
#         me   = last_Me.unsqueeze(0)  + d[:, :, 2:]
#         return traj, me

#     # ── FIX 2: Step-wise direction loss ───────────────────────────────────────
#     def _dir_loss(self, pred_abs, gt_abs, last_pos):
#         """
#         v5: kết hợp overall direction loss + step-wise direction loss.

#         Overall: so hướng tổng thể pred vs gt từ last_pos
#         Step-wise: so hướng velocity từng bước → phạt nặng khi sai hướng di chuyển
#         """
#         # Overall direction (giữ từ v4)
#         ref = last_pos.unsqueeze(0)
#         overall = torch.clamp(
#             0.7 - (F.normalize(gt_abs - ref, p=2, dim=-1) *
#                    F.normalize(pred_abs - ref, p=2, dim=-1)).sum(-1),
#             min=0
#         ).mean()

#         # Step-wise direction: so velocity từng bước
#         if pred_abs.shape[0] >= 2:
#             pred_v = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
#             gt_v   = gt_abs[1:]   - gt_abs[:-1]     # [T-1, B, 2]
#             # Cosine similarity giữa pred velocity và gt velocity
#             cos_sim = (F.normalize(pred_v, dim=-1) *
#                        F.normalize(gt_v,   dim=-1)).sum(-1)  # [T-1, B]
#             # Phạt khi cos < 0.5 (sai hướng > 60°)
#             step_dir = F.relu(0.5 - cos_sim).mean()
#         else:
#             step_dir = pred_abs.new_zeros(1).squeeze()

#         return overall + 2.0 * step_dir   # step_dir weight cao hơn

#     def _smooth_loss(self, traj_abs):
#         """
#         v5: smooth loss trên abs trajectory [T, B, 2].
#         Phạt acceleration lớn (noise/jitter), KHÔNG phạt recurvature dần dần.
#         traj_abs: [T, B, 2]
#         """
#         T = traj_abs.shape[0]
#         if T < 3:
#             return traj_abs.new_zeros(1).squeeze()
#         v   = traj_abs[1:] - traj_abs[:-1]   # velocity  [T-1, B, 2]
#         acc = v[1:] - v[:-1]                  # accel     [T-2, B, 2]
#         return (acc ** 2).mean()

#     def _curvature_loss(self, pred_abs, gt_abs):
#         if pred_abs.shape[0] < 3:
#             return pred_abs.new_zeros(1).squeeze()
#         pred_v    = pred_abs[1:] - pred_abs[:-1]
#         gt_v      = gt_abs[1:]   - gt_abs[:-1]
#         pred_curl = pred_v[1:,:,0]*pred_v[:-1,:,1] - pred_v[1:,:,1]*pred_v[:-1,:,0]
#         gt_curl   = gt_v[1:,:,0] *gt_v[:-1,:,1]   - gt_v[1:,:,1] *gt_v[:-1,:,0]
#         return (F.relu(-(pred_curl * gt_curl)).mean()
#                 + 0.3 * F.mse_loss(pred_curl, gt_curl))

#     def _weighted_disp_loss(self, pred_abs, gt_abs):
#         T = pred_abs.shape[0]
#         w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
#         return (w * (pred_abs - gt_abs).abs()).mean()

#     def _ns_pinn_loss(self, pred_abs):
#         T = pred_abs.shape[0]
#         if T < 4:
#             return pred_abs.new_zeros(1).squeeze()

#         v        = pred_abs[1:] - pred_abs[:-1]
#         zeta     = (v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0])
#         dzeta_dt = zeta[1:] - zeta[:-1]
#         lat_rad  = torch.deg2rad(pred_abs[2:-1,:,1] * 50.0)
#         beta_n   = 2 * OMEGA * lat_rad.cos() * DT_6H
#         v_y_n    = v[1:-1,:,1]
#         residual = dzeta_dt + beta_n * v_y_n
#         return (residual ** 2).mean()

#     def get_loss(self, batch_list):
#         traj_gt = batch_list[1]
#         Me_gt   = batch_list[8]
#         obs     = batch_list[0]
#         obs_Me  = batch_list[7]

#         B      = traj_gt.shape[1]
#         device = traj_gt.device
#         lp, lm = obs[-1], obs_Me[-1]
#         sm     = self.sigma_min

#         x1 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)
#         x0 = torch.randn_like(x1) * sm

#         t     = torch.rand(B, device=device)
#         t_exp = t.view(B, 1, 1)

#         x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
#         denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
#         target_vel = (x1 - (1 - sm) * x_t) / denom

#         pred_vel = self.net(x_t, t, batch_list)
#         fm_loss  = F.mse_loss(pred_vel, target_vel)

#         pred_x1     = x_t + denom * pred_vel
#         pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

#         # v5: dir_loss bao gồm cả step-wise direction
#         dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
#         smt_l  = self._smooth_loss(pred_abs)        # smooth trên abs trajectory
#         disp_l = self._weighted_disp_loss(pred_abs, traj_gt)
#         curv_l = self._curvature_loss(pred_abs, traj_gt)
#         pinn_l = self._ns_pinn_loss(pred_abs)

#         return (fm_loss
#                 + 2.0 * dir_l     # dir_l đã bao gồm step_dir × 2.0
#                 + 0.3 * smt_l
#                 + 1.0 * disp_l
#                 + 1.5 * curv_l
#                 + 0.5 * pinn_l)

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=5, ddim_steps=10):
#         obs_t, obs_m = batch_list[0], batch_list[7]
#         lp, lm       = obs_t[-1], obs_m[-1]
#         device       = lp.device
#         B            = lp.shape[0]
#         dt           = 1.0 / ddim_steps

#         trajs = []
#         for _ in range(num_ensemble):
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
#             for i in range(ddim_steps):
#                 t_b = torch.full((B,), i * dt, device=device)
#                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
#                 x_t[:, :, :2] = x_t[:, :, :2].clamp(-2.0, 2.0)
#             traj, _ = self.rel_to_abs(x_t, lp, lm)
#             trajs.append(traj)

#         final_traj = torch.stack(trajs).mean(0)

#         mes = []
#         for _ in range(max(1, num_ensemble // 2)):
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
#             for i in range(ddim_steps):
#                 t_b = torch.full((B,), i * dt, device=device)
#                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
#                 x_t[:, :, :2] = x_t[:, :, :2].clamp(-2.0, 2.0)
#             _, me = self.rel_to_abs(x_t, lp, lm)
#             mes.append(me)

#         return final_traj, torch.stack(mes).mean(0)


# TCDiffusion = TCFlowMatching

# """
# TCNM/flow_matching_model.py  ── v6  Fix triệt để: gom cụm + sai hướng + đường thẳng
# ======================================================================================

# PHÂN TÍCH 3 VẤN ĐỀ VÀ FIX:

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VẤN ĐỀ 1: GOM CỤM (predict displacement quá nhỏ)
# ────────────────────────────────────────────────────────────────────────
#   Nguyên nhân: x1 = absolute offset từ last_pos, scale ~0.5-3.0
#                Noise x0 = randn * sigma_min = randn * 0.001 (quá nhỏ)
#                OT-CFM path từ x0≈0 đến x1 lớn → velocity field phải
#                học predict vector rất lớn → underestimate → gom cụm

#   Fix: sigma_min 0.001 → 0.01
#        x0 có scale gần x1 hơn → OT-CFM path ngắn hơn → dễ học hơn

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VẤN ĐỀ 2: SAI HƯỚNG (dir_loss collapse về 0)
# ────────────────────────────────────────────────────────────────────────
#   Nguyên nhân: F.normalize(near-zero vector) → unstable gradient
#                Model tìm shortcut predict nhỏ → cos_sim ngẫu nhiên
#                → dir_loss thấp mà không học hướng thật

#   Fix:
#   (a) norm guard: clamp > 1e-3 trước normalize
#   (b) mask: chỉ tính khi gt_norm > 0.02
#   (c) tách overall_dir và step_dir thành 2 loss riêng

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VẤN ĐỀ 3: ĐƯỜNG THẲNG (không học được recurvature)
# ────────────────────────────────────────────────────────────────────────
#   Nguyên nhân: curv_loss weight 1.5 quá thấp
#                Không có loss nào phạt khi pred đi thẳng mà gt rẽ

#   Fix: thay curv_loss bằng heading_change_loss (mạnh hơn)
#        = MSE(signed_curvature_pred, signed_curvature_gt)
#        + relu penalty khi đổi hướng ngược chiều
#        weight 2.0 → ưu tiên học pattern rẽ hướng

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Giữ nguyên: VelocityField (LSTM, UNet3D, Env_net, Transformer),
#             traj_to_rel/rel_to_abs (absolute offset từ v5),
#             OT-CFM training, PINN vorticity, sampling loop
# """
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from TCNM.Unet3D_merge_tiny import Unet3D
# from TCNM.env_net_transformer_gphsplit import Env_net


# # ── Physical constants ────────────────────────────────────────────────────────
# OMEGA      = 7.2921e-5
# R_EARTH    = 6.371e6
# DT_6H      = 6 * 3600
# NORM_TO_MS = 555e3 / DT_6H


# # ── Velocity Field Network (giữ nguyên hoàn toàn từ v4/v5) ───────────────────
# class VelocityField(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=128, sigma_min=0.01):
#         super().__init__()
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.sigma_min = sigma_min

#         self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
#         self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
#         self.obs_lstm     = nn.LSTM(input_size=4, hidden_size=128,
#                                     num_layers=3, batch_first=True, dropout=0.2)
#         self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
#         self.ctx_fc1      = nn.Linear(16 + 64 + 128, 512)
#         self.ctx_ln       = nn.LayerNorm(512)
#         self.ctx_drop     = nn.Dropout(0.15)
#         self.ctx_fc2      = nn.Linear(512, ctx_dim)

#         self.time_fc1   = nn.Linear(128, 256)
#         self.time_fc2   = nn.Linear(256, 128)
#         self.traj_embed = nn.Linear(4, 128)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=128, nhead=8, dim_feedforward=512,
#             dropout=0.15, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
#         self.out_fc1 = nn.Linear(128, 256)
#         self.out_fc2 = nn.Linear(256, 4)

#     def _time_emb(self, t, dim=128):
#         device = t.device
#         half   = dim // 2
#         freq   = torch.exp(
#             torch.arange(half, dtype=torch.float32, device=device)
#             * (-math.log(10000.0) / (half - 1))
#         )
#         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return F.pad(emb, (0, 1)) if dim % 2 else emb

#     def _extract_context(self, batch_list):
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]

#         f_s = self.spatial_enc(image_obs).mean(dim=2)
#         f_s = self.spatial_pool(f_s).flatten(1)
#         f_e, _, _ = self.env_enc(env_data, image_obs)

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         _, (h_n, _) = self.obs_lstm(obs_in)
#         f_h = h_n[-1]

#         ctx = torch.cat([f_s, f_e, f_h], dim=-1)
#         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
#         ctx = self.ctx_drop(ctx)
#         ctx = self.ctx_fc2(ctx)
#         return ctx

#     def forward(self, x_t, t, batch_list):
#         ctx   = self._extract_context(batch_list)
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)
#         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
#         out    = self.transformer(x_emb, memory)
#         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # ── TCFlowMatching v6 ─────────────────────────────────────────────────────────
# class TCFlowMatching(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, num_steps=100,
#                  sigma_min=0.01, **kwargs):   # FIX 1: 0.001 → 0.01
#         super().__init__()
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.sigma_min = sigma_min
#         self.net       = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

#     # ── Encoding: absolute offset từ last_pos (giữ từ v5) ────────────────────
#     @staticmethod
#     def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
#         """Absolute offset — model học hình dạng toàn bộ trajectory."""
#         traj_norm = traj_gt - last_pos.unsqueeze(0)
#         me_norm   = Me_gt   - last_Me.unsqueeze(0)
#         return torch.cat([traj_norm, me_norm], dim=-1).permute(1, 0, 2)

#     @staticmethod
#     def rel_to_abs(rel, last_pos, last_Me):
#         """Inverse: offset + last_pos. Không cumsum."""
#         d    = rel.permute(1, 0, 2)
#         traj = last_pos.unsqueeze(0) + d[:, :, :2]
#         me   = last_Me.unsqueeze(0)  + d[:, :, 2:]
#         return traj, me

#     # ── FIX 2a: Overall direction ─────────────────────────────────────────────
#     def _overall_dir_loss(self, pred_abs, gt_abs, last_pos):
#         """Hướng tổng thể pred vs gt từ last_pos."""
#         ref = last_pos.unsqueeze(0)
#         return torch.clamp(
#             0.7 - (F.normalize(gt_abs - ref, p=2, dim=-1) *
#                    F.normalize(pred_abs - ref, p=2, dim=-1)).sum(-1),
#             min=0
#         ).mean()

#     # ── FIX 2b: Step-wise direction với norm guard ────────────────────────────
#     def _step_dir_loss(self, pred_abs, gt_abs):
#         """
#         Cosine similarity giữa pred_velocity và gt_velocity từng bước.
#         Norm guard: clamp > 1e-3 → gradient ổn định, không collapse.
#         Mask: chỉ tính khi gt di chuyển > 0.02 (≈11 km).
#         """
#         if pred_abs.shape[0] < 2:
#             return pred_abs.new_zeros(1).squeeze()

#         pred_v = pred_abs[1:] - pred_abs[:-1]
#         gt_v   = gt_abs[1:]   - gt_abs[:-1]

#         pred_norm = pred_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
#         gt_norm   = gt_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
#         cos_sim   = ((pred_v / pred_norm) * (gt_v / gt_norm)).sum(-1)

#         mask = (gt_v.norm(dim=-1) > 0.02).float()
#         return (F.relu(0.5 - cos_sim) * mask).sum() / (mask.sum() + 1e-6)

#     # ── FIX 3: Heading change loss (fix đường thẳng + recurvature) ───────────
#     def _heading_change_loss(self, pred_abs, gt_abs):
#         """
#         Phạt khi pred không học được pattern đổi hướng của gt.

#         signed_curvature[t] = cross(v[t+1], v[t]) / (|v[t+1]| * |v[t]|)
#                             ∈ [-1, 1]  (dương = rẽ trái, âm = rẽ phải)

#         Loss = MSE(pred_curv, gt_curv)           ← học magnitude đổi hướng
#              + relu(-(pred_curv * gt_curv)).mean() ← phạt đổi hướng ngược chiều

#         Tại sao tốt hơn curv_loss cũ:
#           - curv_loss cũ: relu(-(pred_curl * gt_curl)) chỉ phạt dấu sai
#           - heading_change: thêm MSE → học được cả magnitude của recurvature
#         """
#         if pred_abs.shape[0] < 3:
#             return pred_abs.new_zeros(1).squeeze()

#         pred_v = pred_abs[1:] - pred_abs[:-1]
#         gt_v   = gt_abs[1:]   - gt_abs[:-1]

#         def signed_curv(v):
#             cross = v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0]
#             n1 = v[1:].norm(dim=-1).clamp(min=1e-3)
#             n2 = v[:-1].norm(dim=-1).clamp(min=1e-3)
#             return cross / (n1 * n2)

#         pred_curv = signed_curv(pred_v)
#         gt_curv   = signed_curv(gt_v)

#         mse_l  = F.mse_loss(pred_curv, gt_curv)
#         sign_l = F.relu(-(pred_curv * gt_curv)).mean()
#         return mse_l + sign_l

#     def _smooth_loss(self, traj_abs):
#         """Phạt acceleration lớn (jitter). Không phạt recurvature dần dần."""
#         T = traj_abs.shape[0]
#         if T < 3:
#             return traj_abs.new_zeros(1).squeeze()
#         v   = traj_abs[1:] - traj_abs[:-1]
#         acc = v[1:] - v[:-1]
#         return (acc ** 2).mean()

#     def _weighted_disp_loss(self, pred_abs, gt_abs):
#         T = pred_abs.shape[0]
#         w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
#         return (w * (pred_abs - gt_abs).abs()).mean()

#     def _ns_pinn_loss(self, pred_abs):
#         T = pred_abs.shape[0]
#         if T < 4:
#             return pred_abs.new_zeros(1).squeeze()
#         v        = pred_abs[1:] - pred_abs[:-1]
#         zeta     = (v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0])
#         dzeta_dt = zeta[1:] - zeta[:-1]
#         lat_rad  = torch.deg2rad(pred_abs[2:-1,:,1] * 50.0)
#         beta_n   = 2 * OMEGA * lat_rad.cos() * DT_6H
#         v_y_n    = v[1:-1,:,1]
#         residual = dzeta_dt + beta_n * v_y_n
#         return (residual ** 2).mean()

#     def get_loss(self, batch_list):
#         """
#         v6 Loss breakdown:
#           1.0 * fm_loss       OT-CFM flow matching
#           1.5 * overall_dir   hướng tổng thể từ last_pos
#           1.5 * step_dir      hướng từng bước (norm guard + mask)
#           1.0 * disp_l        weighted displacement (xa hơn phạt nặng hơn)
#           2.0 * heading_l     pattern đổi hướng = FIX chính cho recurvature
#           0.2 * smooth_l      anti-jitter
#           0.5 * pinn_l        vorticity constraint
#         """
#         traj_gt = batch_list[1]
#         Me_gt   = batch_list[8]
#         obs     = batch_list[0]
#         obs_Me  = batch_list[7]

#         B      = traj_gt.shape[1]
#         device = traj_gt.device
#         lp, lm = obs[-1], obs_Me[-1]
#         sm     = self.sigma_min

#         x1 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)
#         x0 = torch.randn_like(x1) * sm   # FIX 1: sm=0.01

#         t     = torch.rand(B, device=device)
#         t_exp = t.view(B, 1, 1)

#         x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
#         denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
#         target_vel = (x1 - (1 - sm) * x_t) / denom

#         pred_vel = self.net(x_t, t, batch_list)
#         fm_loss  = F.mse_loss(pred_vel, target_vel)

#         pred_x1     = x_t + denom * pred_vel
#         pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

#         overall_dir = self._overall_dir_loss(pred_abs, traj_gt, lp)
#         step_dir    = self._step_dir_loss(pred_abs, traj_gt)
#         disp_l      = self._weighted_disp_loss(pred_abs, traj_gt)
#         heading_l   = self._heading_change_loss(pred_abs, traj_gt)
#         smooth_l    = self._smooth_loss(pred_abs)
#         pinn_l      = self._ns_pinn_loss(pred_abs)

#         return (  1.0 * fm_loss
#                 + 1.5 * overall_dir
#                 + 1.5 * step_dir
#                 + 1.0 * disp_l
#                 + 2.0 * heading_l
#                 + 0.2 * smooth_l
#                 + 0.5 * pinn_l)

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=5, ddim_steps=10):
#         obs_t, obs_m = batch_list[0], batch_list[7]
#         lp, lm       = obs_t[-1], obs_m[-1]
#         device       = lp.device
#         B            = lp.shape[0]
#         dt           = 1.0 / ddim_steps

#         trajs = []
#         for _ in range(num_ensemble):
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
#             for i in range(ddim_steps):
#                 t_b = torch.full((B,), i * dt, device=device)
#                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
#                 x_t[:, :, :2] = x_t[:, :, :2].clamp(-3.0, 3.0)
#             traj, _ = self.rel_to_abs(x_t, lp, lm)
#             trajs.append(traj)

#         final_traj = torch.stack(trajs).mean(0)

#         mes = []
#         for _ in range(max(1, num_ensemble // 2)):
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
#             for i in range(ddim_steps):
#                 t_b = torch.full((B,), i * dt, device=device)
#                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
#                 x_t[:, :, :2] = x_t[:, :, :2].clamp(-3.0, 3.0)
#             _, me = self.rel_to_abs(x_t, lp, lm)
#             mes.append(me)

#         return final_traj, torch.stack(mes).mean(0)


# TCDiffusion = TCFlowMatching

"""
TCNM/flow_matching_model.py  ── v7
====================================
OT-CFM flow matching + physics-informed losses for TC trajectory prediction.
Loss function follows Eq. (60) exactly as written in the report.

Loss components:
    L1  FM loss          weight 1.0   Eq. (61) — OT-CFM velocity matching
    L2  overall_dir      weight 2.0   Eq. (62) — cosine of net displacement
    L3  step_dir         weight 0.5   Eq. (63) — per-step direction cosine
    L4  displacement     weight 1.0   Eq. (64) — weighted position L1
    L5  heading_change   weight 2.0   Eq. (65-66) — signed curvature MSE
    L6  smoothness       weight 0.2   Eq. (67) — discrete acceleration L2
    L7  PINN (BVE)       weight 0.5   Eq. (43-45) — barotropic vorticity eq.

PINN implementation notes
──────────────────────────
Full BVE per Eq. (44):  r_k = ∂ζ/∂t + u_k·∂ζ/∂x + v_k·∂ζ/∂y ≈ 0

ERA5 is available at the 8 *observed* steps only (batch_list[13]).
For the 12 *predicted* steps we use the last observed ERA5 patch as a
temporal approximation.  Steering flow changes ≈2–5 % per 6 h (synoptic
scale), which is acceptable for soft regularisation.

If ERA5 u/v at 850 hPa cannot be parsed from batch_list[13], the code
falls back to the simplified BVE (∂ζ/∂t ≈ 0) in normalised coordinates.

ERA5 dict format expected (batch_list[13]):
    key 'u850' → tensor [B, T_obs, H, W]   zonal wind (m s⁻¹)
    key 'v850' → tensor [B, T_obs, H, W]   meridional wind (m s⁻¹)
    Grid: 9°×9° centred on storm at each obs step, 0.25°/pixel.
    Accepted alternative: tensor [B, C, T_obs, H, W] with channel
    order [u200,u500,u850,v200,v500,v850,...] (u850=ch2, v850=ch5).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net

# ── Physical constants ─────────────────────────────────────────────────────────
OMEGA    = 7.2921e-5   # Earth rotation rate  (rad s⁻¹)
R_EARTH  = 6.371e6     # Earth radius         (m)
DT_6H    = 6 * 3600    # Timestep             (s)

# Normalisation: lon_norm*5+180 = lon_deg,  lat_norm*5 = lat_deg
NORM_TO_DEG = 5.0

# ERA5 grid geometry
ERA5_RES_DEG  = 0.25   # degrees per pixel
DELTA_DEG     = 0.10   # perturbation for finite-difference vorticity (≈11 km)

NORM_TO_M        = NORM_TO_DEG * 111000.0
BETA_NORM_FACTOR = 2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH   # ~0.276

# ── ERA5 helpers ───────────────────────────────────────────────────────────────

def _bilinear_interp(
    field:      torch.Tensor,   # [B, H, W]
    center_lon: torch.Tensor,   # [B]  grid-centre longitude (degrees)
    center_lat: torch.Tensor,   # [B]  grid-centre latitude  (degrees)
    query_lon:  torch.Tensor,   # [B, N]
    query_lat:  torch.Tensor,   # [B, N]
) -> torch.Tensor:              # [B, N]
    """
    Differentiable bilinear interpolation of an ERA5 patch.

    Grid convention
    ───────────────
    Pixel (0,0) is top-left  → (lat+HALF, lon-HALF).
    Row index increases southward; column index increases eastward.
    grid_sample uses (x=col, y=row) normalised to [-1, 1].
    """
    H, W = field.shape[-2], field.shape[-1]
    half_lon = (W // 2) * ERA5_RES_DEG
    half_lat = (H // 2) * ERA5_RES_DEG

    dlon = query_lon - center_lon.unsqueeze(1)   # [B, N]
    dlat = query_lat - center_lat.unsqueeze(1)   # [B, N]

    gx = ( dlon / half_lon).clamp(-1.0, 1.0)    # east  → positive x
    gy = (-dlat / half_lat).clamp(-1.0, 1.0)    # north → negative y (row 0 = top)

    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)   # [B, 1, N, 2]
    out  = F.grid_sample(
        field.unsqueeze(1),                              # [B, 1, H, W]
        grid,
        mode='bilinear', padding_mode='border', align_corners=True,
    )
    return out.squeeze(1).squeeze(1)                     # [B, N]


def _vorticity_era5(
    u850: torch.Tensor,   # [B, H, W]
    v850: torch.Tensor,   # [B, H, W]
    clon: torch.Tensor,   # [B]
    clat: torch.Tensor,   # [B]
    lon:  torch.Tensor,   # [B, N]  query longitudes (degrees)
    lat:  torch.Tensor,   # [B, N]  query latitudes  (degrees)
) -> torch.Tensor:        # [B, N]
    """
    Relative vorticity at query points via centred finite difference (Eq. 43):
        ζ = ∂v/∂x − ∂u/∂y
          ≈ [v(λ+δ)−v(λ−δ)] / (2 δx)  −  [u(ϕ+δ)−u(ϕ−δ)] / (2 δy)

    δx = R⊕ cos(ϕ) δ π/180,   δy = R⊕ δ π/180
    """
    delta_m = DELTA_DEG * math.pi / 180.0 * R_EARTH          # scalar (m)

    v_xp = _bilinear_interp(v850, clon, clat, lon + DELTA_DEG, lat)
    v_xm = _bilinear_interp(v850, clon, clat, lon - DELTA_DEG, lat)
    u_yp = _bilinear_interp(u850, clon, clat, lon, lat + DELTA_DEG)
    u_ym = _bilinear_interp(u850, clon, clat, lon, lat - DELTA_DEG)

    cos_lat = torch.cos(torch.deg2rad(lat))                   # [B, N]
    dx = (cos_lat * delta_m).clamp(min=1.0)                   # [B, N] (m)
    dy = delta_m                                               # scalar (m)

    return (v_xp - v_xm) / (2.0 * dx) - (u_yp - u_ym) / (2.0 * dy)


def _parse_era5_uv850(
    batch_list: List,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
           torch.Tensor, torch.Tensor, bool]:
    """
    Extract u850, v850 at the last observed timestep.

    Returns
    -------
    u850_last : [B, H, W] or None
    v850_last : [B, H, W] or None
    clon      : [B]  storm longitude at last obs (degrees)
    clat      : [B]  storm latitude  at last obs (degrees)
    ok        : bool  True if ERA5 was parsed successfully
    """
    obs_traj = batch_list[0]            # [T_obs, B, 2]  normalised
    last     = obs_traj[-1]             # [B, 2]
    clon     = last[:, 0] * NORM_TO_DEG + 180.0
    clat     = last[:, 1] * NORM_TO_DEG

    env = batch_list[13]

    # Format A: dict with variable keys
    if isinstance(env, dict):
        u_key = next((k for k in ('u850', 'U850', 'u_850') if k in env), None)
        v_key = next((k for k in ('v850', 'V850', 'v_850') if k in env), None)
        if u_key is None or v_key is None:
            return None, None, clon, clat, False
        u = env[u_key]
        v = env[v_key]
        # Shape [B, T_obs, H, W] or [B, H, W]
        u_last = u[:, -1] if u.dim() == 4 else u
        v_last = v[:, -1] if v.dim() == 4 else v
        return u_last, v_last, clon, clat, True

    # Format B: tensor [B, C, T_obs, H, W]  (ch2=u850, ch5=v850)
    if isinstance(env, torch.Tensor) and env.dim() == 5:
        return env[:, 2, -1], env[:, 5, -1], clon, clat, True

    # Format C: tensor [B, C, H, W]
    if isinstance(env, torch.Tensor) and env.dim() == 4:
        return env[:, 2], env[:, 5], clon, clat, True

    return None, None, clon, clat, False


# ── Velocity Field ─────────────────────────────────────────────────────────────

class VelocityField(nn.Module):
    """
    Learned OT-CFM velocity field conditioned on observed track + ERA5 context.

    Architecture
    ────────────
    • UNet3D      – spatial encoder for ERA5 image patches
    • Env_net     – multi-level environmental context encoder
    • LSTM        – observed track encoder
    • TransformerDecoder – generates per-step velocity for pred_len steps
    """

    def __init__(
        self,
        pred_len:  int   = 12,
        obs_len:   int   = 8,
        ctx_dim:   int   = 128,
        sigma_min: float = 0.02,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        # Encoders
        self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
        self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
        self.obs_lstm     = nn.LSTM(
            input_size=4, hidden_size=128, num_layers=3,
            batch_first=True, dropout=0.2,
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Context projection
        self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        # Time embedding projection
        self.time_fc1 = nn.Linear(128, 256)
        self.time_fc2 = nn.Linear(256, 128)

        # Trajectory token embedding + learnable positional encoding
        self.traj_embed = nn.Linear(4, 128)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

        # Decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=128, nhead=8, dim_feedforward=512,
                dropout=0.15, activation='gelu', batch_first=True,
            ),
            num_layers=4,
        )
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

    # ------------------------------------------------------------------
    def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10000.0) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, 1)) if dim % 2 else emb

    def _context(self, batch_list: List) -> torch.Tensor:
        obs_traj  = batch_list[0]     # [T_obs, B, 2]
        obs_Me    = batch_list[7]     # [T_obs, B, 2]
        image_obs = batch_list[11]    # [B, 1, T_obs, H, W]
        env_data  = batch_list[13]

        f_s = self.spatial_enc(image_obs).mean(dim=2)   # [B, C, H, W]
        f_s = self.spatial_pool(f_s).flatten(1)         # [B, 16]

        f_e, _, _ = self.env_enc(env_data, image_obs)   # [B, 64]

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B, T_obs, 4]
        _, (h_n, _) = self.obs_lstm(obs_in)
        f_h = h_n[-1]                                   # [B, 128]

        ctx = torch.cat([f_s, f_e, f_h], dim=-1)        # [B, 208]
        ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
        ctx = self.ctx_drop(ctx)
        return self.ctx_fc2(ctx)                         # [B, ctx_dim]

    def forward(
        self,
        x_t:       torch.Tensor,   # [B, T_pred, 4]
        t:         torch.Tensor,   # [B]
        batch_list: List,
    ) -> torch.Tensor:             # [B, T_pred, 4]
        ctx   = self._context(batch_list)                # [B, 128]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)                     # [B, 128]

        x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)  # [B, T, 128]
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)   # [B, 2, 128]

        out = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))   # [B, T, 4]


# ── TCFlowMatching ─────────────────────────────────────────────────────────────

class TCFlowMatching(nn.Module):
    """
    Tropical cyclone trajectory prediction via Optimal-Transport CFM
    with physics-informed auxiliary losses.

    Loss weights follow Eq. (60) exactly.
    """

    def __init__(
        self,
        pred_len:  int   = 12,
        obs_len:   int   = 8,
        sigma_min: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min
        self.net       = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

    # ── Coordinate encoding ────────────────────────────────────────────────────

    @staticmethod
    def traj_to_rel(
        traj_gt:  torch.Tensor,   # [T, B, 2]
        Me_gt:    torch.Tensor,   # [T, B, 2]
        last_pos: torch.Tensor,   # [B, 2]
        last_Me:  torch.Tensor,   # [B, 2]
    ) -> torch.Tensor:            # [B, T, 4]
        """Encode as absolute offset from last observed position."""
        traj_enc = traj_gt - last_pos.unsqueeze(0)
        me_enc   = Me_gt   - last_Me.unsqueeze(0)
        return torch.cat([traj_enc, me_enc], dim=-1).permute(1, 0, 2)

    @staticmethod
    def rel_to_abs(
        rel:      torch.Tensor,   # [B, T, 4]
        last_pos: torch.Tensor,   # [B, 2]
        last_Me:  torch.Tensor,   # [B, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode offset encoding → absolute normalised coordinates."""
        d    = rel.permute(1, 0, 2)                  # [T, B, 4]
        traj = last_pos.unsqueeze(0) + d[:, :, :2]   # [T, B, 2]
        me   = last_Me.unsqueeze(0)  + d[:, :, 2:]   # [T, B, 2]
        return traj, me

    # ── L2: Overall direction loss — Eq. (62) ─────────────────────────────────

    def _overall_dir_loss(
        self,
        pred_abs: torch.Tensor,   # [T, B, 2]
        gt_abs:   torch.Tensor,   # [T, B, 2]
        last_pos: torch.Tensor,   # [B, 2]
    ) -> torch.Tensor:
        """
        L = 1 − cos( x̂_k − x₀,  x_k − x₀ )   averaged over k=1..T
        """
        ref  = last_pos.unsqueeze(0)

        p = pred_abs - ref
        g = gt_abs   - ref

        pn = p.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return (1.0 - ((p / pn) * (g / gn)).sum(-1)).mean()

    # ── L3: Step-wise direction loss — Eq. (63) ───────────────────────────────

    def _step_dir_loss(
        self,
        pred_abs: torch.Tensor,
        gt_abs:   torch.Tensor,
    ) -> torch.Tensor:
        """
        L = mean_k( 1 − cos( v̂_k, v_k ) )   where v_k = x_{k+1} − x_k
        """
        if pred_abs.shape[0] < 2:
            return pred_abs.new_zeros(())

        pv = pred_abs[1:] - pred_abs[:-1]
        gv = gt_abs[1:]   - gt_abs[:-1]

        pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()

    # ── L4: Displacement loss — Eq. (64) ──────────────────────────────────────

    def _disp_loss(
        self,
        pred_abs: torch.Tensor,
        gt_abs:   torch.Tensor,
    ) -> torch.Tensor:
        """Weighted L1 displacement; weight increases toward later steps."""
        T = pred_abs.shape[0]
        w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
        return (w * (pred_abs - gt_abs).abs()).mean()

    # ── L5: Heading change loss — Eq. (65–66) ─────────────────────────────────

    def _heading_loss(
        self,
        pred_abs: torch.Tensor,
        gt_abs:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Signed curvature:  κ_k = cross(v_{k+1}, v_k) / (|v_{k+1}| |v_k|)

        L = mean_k[ (κ̂_k − κ_k)²  +  ReLU(−κ̂_k · κ_k) ]

        The ReLU term penalises wrong-direction turns.
        """
        if pred_abs.shape[0] < 3:
            return pred_abs.new_zeros(())

        pv = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
        gv = gt_abs[1:]   - gt_abs[:-1]

        def curvature(v: torch.Tensor) -> torch.Tensor:
            # cross(v[t+1], v[t]) = vx[t+1]*vy[t] − vy[t+1]*vx[t]
            cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
            n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
            n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
            return cross / (n1 * n2)                         # [T-2, B]

        pc = curvature(pv)
        gc = curvature(gv)

        return F.mse_loss(pc, gc) + F.relu(-(pc * gc)).mean()

    # ── L6: Smoothness loss — Eq. (67) ────────────────────────────────────────

    def _smooth_loss(self, traj_abs: torch.Tensor) -> torch.Tensor:
        """
        L = mean_k ‖x_{k+1} − 2x_k + x_{k−1}‖²
        Discrete second derivative (acceleration) — zero for straight lines.
        """
        if traj_abs.shape[0] < 3:
            return traj_abs.new_zeros(())
        acc = traj_abs[2:] - 2.0 * traj_abs[1:-1] + traj_abs[:-2]
        return (acc ** 2).mean()

    # ── L7: PINN (BVE) loss — Eq. (43–45) ─────────────────────────────────────

    def _pinn_loss(
        self,
        pred_abs:  torch.Tensor,   # [T, B, 2]  normalised
        batch_list: List,
    ) -> torch.Tensor:
        """
        Barotropic Vorticity Equation residual:
            r_k = ∂ζ/∂t  +  u_k · ∂ζ/∂x  +  v_k · ∂ζ/∂y  ≈ 0

        u_k, v_k — storm velocity from track  (centered diff, Eq. 40–41)
        ζ        — ERA5 relative vorticity at storm position  (Eq. 43)
        ∂ζ/∂t    — centred time difference of ζ  (Eq. 44a)
        ∂ζ/∂x/y  — centred spatial difference of ζ  (Eq. 44b–c)
        """
        T = pred_abs.shape[0]
        if T < 4:
            return pred_abs.new_zeros(())

        # Degrees
        lon = pred_abs[..., 0] * NORM_TO_DEG + 180.0   # [T, B]
        lat = pred_abs[..., 1] * NORM_TO_DEG            # [T, B]

        # ── Storm velocity u_k, v_k (m s⁻¹) — interior k=1..T-2 ──────────────
        dlon_rad = (lon[2:] - lon[:-2]) * (math.pi / 180.0)   # [T-2, B]
        dlat_rad = (lat[2:] - lat[:-2]) * (math.pi / 180.0)   # [T-2, B]
        cos_lat  = torch.cos(torch.deg2rad(lat[1:-1]))         # [T-2, B]

        u_k = R_EARTH * cos_lat * dlon_rad / (2.0 * DT_6H)    # [T-2, B]
        v_k = R_EARTH           * dlat_rad / (2.0 * DT_6H)    # [T-2, B]

        # ── Try ERA5 ──────────────────────────────────────────────────────────
        u850, v850, clon, clat, era5_ok = _parse_era5_uv850(batch_list)

        if not era5_ok:
            return self._pinn_loss_simplified(pred_abs)

        device = pred_abs.device
        u850 = u850.to(device)
        v850 = v850.to(device)
        clon = clon.to(device)
        clat = clat.to(device)

        # ── ζ at every predicted step — [B, T] then → [T, B] ─────────────────
        lon_BT = lon.permute(1, 0)   # [B, T]
        lat_BT = lat.permute(1, 0)   # [B, T]

        zeta_BT = _vorticity_era5(u850, v850, clon, clat, lon_BT, lat_BT)
        zeta    = zeta_BT.permute(1, 0)   # [T, B]

        # ── ∂ζ/∂t  (centred, s⁻²) ─────────────────────────────────────────────
        # Interior k=1..T-2
        dzeta_dt = (zeta[2:] - zeta[:-2]) / (2.0 * DT_6H)    # [T-2, B]

        # ── ∂ζ/∂x, ∂ζ/∂y  (centred spatial diff, m⁻¹ s⁻¹) ──────────────────
        lon_int = lon_BT[:, 1:-1]   # [B, T-2]
        lat_int = lat_BT[:, 1:-1]   # [B, T-2]
        delta_m = DELTA_DEG * math.pi / 180.0 * R_EARTH        # (m)

        zeta_xp = _vorticity_era5(u850, v850, clon, clat,
                                   lon_int + DELTA_DEG, lat_int).permute(1, 0)  # [T-2, B]
        zeta_xm = _vorticity_era5(u850, v850, clon, clat,
                                   lon_int - DELTA_DEG, lat_int).permute(1, 0)
        zeta_yp = _vorticity_era5(u850, v850, clon, clat,
                                   lon_int, lat_int + DELTA_DEG).permute(1, 0)
        zeta_ym = _vorticity_era5(u850, v850, clon, clat,
                                   lon_int, lat_int - DELTA_DEG).permute(1, 0)

        cos_int = torch.cos(torch.deg2rad(lat[1:-1]))           # [T-2, B]
        dx = (cos_int * delta_m).clamp(min=1.0)                 # [T-2, B]
        dy = delta_m

        dzeta_dx = (zeta_xp - zeta_xm) / (2.0 * dx)            # [T-2, B]
        dzeta_dy = (zeta_yp - zeta_ym) / (2.0 * dy)            # [T-2, B]

        # ── BVE residual  r_k = ∂ζ/∂t + u·∂ζ/∂x + v·∂ζ/∂y — Eq. (44) ───────
        residual = dzeta_dt + u_k * dzeta_dx + v_k * dzeta_dy   # [T-2, B]

        # ── L_PINN = (1/N) Σ r_k²  — Eq. (45) ───────────────────────────────
        return (residual ** 2).mean()

    def _pinn_loss_simplified(self, pred_abs: torch.Tensor) -> torch.Tensor:
        """
        Fallback when ERA5 u/v850 is unavailable.
        Simplified BVE  ∂ζ/∂t + β v ≈ 0  in normalised coordinates.
        Provides non-zero, well-scaled gradient without ERA5 data.
        """

        T = pred_abs.shape[0]
        if T < 4:
            return pred_abs.new_zeros(1).squeeze()

        v  = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
        vx = v[..., 0]                       # [T-1, B]
        vy = v[..., 1]                       # [T-1, B]

        # ζ[t] = vx[t+1]*vy[t] - vy[t+1]*vx[t]  →  [T-2, B]
        zeta = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]

        if zeta.shape[0] < 2:
            return pred_abs.new_zeros(1).squeeze()

        # Δζ[t] = ζ[t+1] - ζ[t]  →  [T-3, B]
        dzeta = zeta[1:] - zeta[:-1]

        # β at positions 2..T-2  →  [T-3, B]
        # pred_abs[2 : T-1] has shape [T-3, B, 2]
        lat_norm = pred_abs[2:T-1, :, 1]                          # [T-3, B]
        lat_rad  = lat_norm * NORM_TO_DEG * (math.pi / 180.0)
        beta_n   = BETA_NORM_FACTOR * torch.cos(lat_rad)          # [T-3, B]

        # v_y at same positions  →  [T-3, B]
        # v[t] = pred_abs[t+1] - pred_abs[t], so v[1:T-2] aligns with steps 2..T-2
        v_y = vy[1:T-2]                                           # [T-3, B]

        residual = dzeta + beta_n * v_y                           # [T-3, B]
        return (residual ** 2).mean()

    # ── Training loss — Eq. (60) ───────────────────────────────────────────────

    def get_loss(self, batch_list: List) -> torch.Tensor:
        """
        L_total = 1.0·L_FM  +  2.0·L_dir  +  0.5·L_step
                + 1.0·L_disp  +  2.0·L_heading  +  0.2·L_smooth
                + 0.5·L_PINN
        """
        traj_gt = batch_list[1]    # [T_pred, B, 2]
        Me_gt   = batch_list[8]    # [T_pred, B, 2]
        obs_t   = batch_list[0]    # [T_obs, B, 2]
        obs_Me  = batch_list[7]    # [T_obs, B, 2]

        B      = traj_gt.shape[1]
        device = traj_gt.device
        lp     = obs_t[-1]         # [B, 2]  last observed position
        lm     = obs_Me[-1]        # [B, 2]
        sm     = self.sigma_min

        # ── OT-CFM interpolation ───────────────────────────────────────────────
        x1    = self.traj_to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
        x0    = torch.randn_like(x1) * sm
        t     = torch.rand(B, device=device)
        te    = t.view(B, 1, 1)

        x_t        = te * x1 + (1.0 - te * (1.0 - sm)) * x0
        denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
        target_vel = (x1 - (1.0 - sm) * x_t) / denom

        pred_vel = self.net(x_t, t, batch_list)
        fm_loss  = F.mse_loss(pred_vel, target_vel)

        # ── Recover predicted absolute trajectory ──────────────────────────────
        pred_x1      = x_t + denom * pred_vel
        pred_abs, _  = self.rel_to_abs(pred_x1, lp, lm)   # [T, B, 2]

        # ── Auxiliary losses ───────────────────────────────────────────────────
        l_dir     = self._overall_dir_loss(pred_abs, traj_gt, lp)
        l_step    = self._step_dir_loss(pred_abs, traj_gt)
        l_disp    = self._disp_loss(pred_abs, traj_gt)
        l_heading = self._heading_loss(pred_abs, traj_gt)
        l_smooth  = self._smooth_loss(pred_abs)
        l_pinn    = self._pinn_loss(pred_abs, batch_list)

        # ── Eq. (60) ───────────────────────────────────────────────────────────
        return (
            1.0 * fm_loss
          + 2.0 * l_dir
          + 0.5 * l_step
          + 1.0 * l_disp
          + 2.0 * l_heading
          + 0.2 * l_smooth
          + 0.5 * l_pinn
        )

    def get_loss_breakdown(self, batch_list: List) -> dict:
        """Same as get_loss() but also returns individual component values for logging."""
        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs_t   = batch_list[0]
        obs_Me  = batch_list[7]

        B, device = traj_gt.shape[1], traj_gt.device
        lp, lm    = obs_t[-1], obs_Me[-1]
        sm        = self.sigma_min

        x1    = self.traj_to_rel(traj_gt, Me_gt, lp, lm)
        x0    = torch.randn_like(x1) * sm
        t     = torch.rand(B, device=device)
        te    = t.view(B, 1, 1)

        # x_t        = te * x1 + (1.0 - te * (1.0 - sm)) * x0
        # denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
        # target_vel = (x1 - (1.0 - sm) * x_t) / denom

        # OT-CFM interpolation (Tong et al. 2024, Eq.51 báo cáo đã cập nhật)
        # x_t = t*x1 + (1 - t*(1-sigma_min))*x0
        # Khác với FM thuần (sigma_min=0): thêm sigma_min để source distribution
        # không collapse, giúp training ổn định với dataset nhỏ (N=2196).
        # sigma_min=0.02 (v7) thay vì 0.001 (v4) để tránh near-deterministic paths.
        x_t   = te * x1 + (1.0 - te * (1.0 - self.sigma_min)) * x0
        denom = (1.0 - (1.0 - self.sigma_min) * te).clamp(min=1e-5)
        # target velocity = (x1 - (1-sigma_min)*x0) / (1 - (1-sigma_min)*t)
        target_vel = (x1 - (1.0 - self.sigma_min) * x0) / denom
        
        pred_vel = self.net(x_t, t, batch_list)
        fm_loss  = F.mse_loss(pred_vel, target_vel)

        pred_x1     = x_t + denom * pred_vel
        pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

        l_dir     = self._overall_dir_loss(pred_abs, traj_gt, lp)
        l_step    = self._step_dir_loss(pred_abs, traj_gt)
        l_disp    = self._disp_loss(pred_abs, traj_gt)
        l_heading = self._heading_loss(pred_abs, traj_gt)
        l_smooth  = self._smooth_loss(pred_abs)
        l_pinn    = self._pinn_loss(pred_abs, batch_list)

        total = (1.0*fm_loss + 2.0*l_dir + 0.5*l_step + 1.0*l_disp
               + 2.0*l_heading + 0.2*l_smooth + 0.5*l_pinn)

        return {
            'total':   total,
            'fm':      fm_loss.item(),
            'dir':     l_dir.item(),
            'step':    l_step.item(),
            'disp':    l_disp.item(),
            'heading': l_heading.item(),
            'smooth':  l_smooth.item(),
            'pinn':    l_pinn.item(),
        }

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list:   List,
        num_ensemble: int = 5,
        ddim_steps:   int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Euler integration of the learned velocity field.

        Returns predicted trajectory [T_pred, B, 2] and intensity [T_pred, B, 2].
        Clamp offset to ±5 normalised units (≈ 2750 km from last position).
        """
        lp     = batch_list[0][-1]    # [B, 2]
        lm     = batch_list[7][-1]    # [B, 2]
        B      = lp.shape[0]
        device = lp.device
        dt     = 1.0 / ddim_steps

        traj_samples = []
        me_samples   = []

        for _ in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2].clamp_(-5.0, 5.0)
            traj, me = self.rel_to_abs(x_t, lp, lm)
            traj_samples.append(traj)
            me_samples.append(me)

        return (
            torch.stack(traj_samples).mean(0),
            torch.stack(me_samples).mean(0),
        )


# Backward-compatibility alias
TCDiffusion = TCFlowMatching