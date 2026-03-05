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

"""
TCNM/flow_matching_model.py  ── v6  Fix triệt để: gom cụm + sai hướng + đường thẳng
======================================================================================

PHÂN TÍCH 3 VẤN ĐỀ VÀ FIX:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VẤN ĐỀ 1: GOM CỤM (predict displacement quá nhỏ)
────────────────────────────────────────────────────────────────────────
  Nguyên nhân: x1 = absolute offset từ last_pos, scale ~0.5-3.0
               Noise x0 = randn * sigma_min = randn * 0.001 (quá nhỏ)
               OT-CFM path từ x0≈0 đến x1 lớn → velocity field phải
               học predict vector rất lớn → underestimate → gom cụm

  Fix: sigma_min 0.001 → 0.01
       x0 có scale gần x1 hơn → OT-CFM path ngắn hơn → dễ học hơn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VẤN ĐỀ 2: SAI HƯỚNG (dir_loss collapse về 0)
────────────────────────────────────────────────────────────────────────
  Nguyên nhân: F.normalize(near-zero vector) → unstable gradient
               Model tìm shortcut predict nhỏ → cos_sim ngẫu nhiên
               → dir_loss thấp mà không học hướng thật

  Fix:
  (a) norm guard: clamp > 1e-3 trước normalize
  (b) mask: chỉ tính khi gt_norm > 0.02
  (c) tách overall_dir và step_dir thành 2 loss riêng

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VẤN ĐỀ 3: ĐƯỜNG THẲNG (không học được recurvature)
────────────────────────────────────────────────────────────────────────
  Nguyên nhân: curv_loss weight 1.5 quá thấp
               Không có loss nào phạt khi pred đi thẳng mà gt rẽ

  Fix: thay curv_loss bằng heading_change_loss (mạnh hơn)
       = MSE(signed_curvature_pred, signed_curvature_gt)
       + relu penalty khi đổi hướng ngược chiều
       weight 2.0 → ưu tiên học pattern rẽ hướng

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Giữ nguyên: VelocityField (LSTM, UNet3D, Env_net, Transformer),
            traj_to_rel/rel_to_abs (absolute offset từ v5),
            OT-CFM training, PINN vorticity, sampling loop
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net


# ── Physical constants ────────────────────────────────────────────────────────
OMEGA      = 7.2921e-5
R_EARTH    = 6.371e6
DT_6H      = 6 * 3600
NORM_TO_MS = 555e3 / DT_6H


# ── Velocity Field Network (giữ nguyên hoàn toàn từ v4/v5) ───────────────────
class VelocityField(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, ctx_dim=128, sigma_min=0.01):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
        self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
        self.obs_lstm     = nn.LSTM(input_size=4, hidden_size=128,
                                    num_layers=3, batch_first=True, dropout=0.2)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.ctx_fc1      = nn.Linear(16 + 64 + 128, 512)
        self.ctx_ln       = nn.LayerNorm(512)
        self.ctx_drop     = nn.Dropout(0.15)
        self.ctx_fc2      = nn.Linear(512, ctx_dim)

        self.time_fc1   = nn.Linear(128, 256)
        self.time_fc2   = nn.Linear(256, 128)
        self.traj_embed = nn.Linear(4, 128)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128, nhead=8, dim_feedforward=512,
            dropout=0.15, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

    def _time_emb(self, t, dim=128):
        device = t.device
        half   = dim // 2
        freq   = torch.exp(
            torch.arange(half, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, 1)) if dim % 2 else emb

    def _extract_context(self, batch_list):
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        f_s = self.spatial_enc(image_obs).mean(dim=2)
        f_s = self.spatial_pool(f_s).flatten(1)
        f_e, _, _ = self.env_enc(env_data, image_obs)

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        _, (h_n, _) = self.obs_lstm(obs_in)
        f_h = h_n[-1]

        ctx = torch.cat([f_s, f_e, f_h], dim=-1)
        ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
        ctx = self.ctx_drop(ctx)
        ctx = self.ctx_fc2(ctx)
        return ctx

    def forward(self, x_t, t, batch_list):
        ctx   = self._extract_context(batch_list)
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)
        x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
        out    = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))


# ── TCFlowMatching v6 ─────────────────────────────────────────────────────────
class TCFlowMatching(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, num_steps=100,
                 sigma_min=0.01, **kwargs):   # FIX 1: 0.001 → 0.01
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min
        self.net       = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

    # ── Encoding: absolute offset từ last_pos (giữ từ v5) ────────────────────
    @staticmethod
    def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
        """Absolute offset — model học hình dạng toàn bộ trajectory."""
        traj_norm = traj_gt - last_pos.unsqueeze(0)
        me_norm   = Me_gt   - last_Me.unsqueeze(0)
        return torch.cat([traj_norm, me_norm], dim=-1).permute(1, 0, 2)

    @staticmethod
    def rel_to_abs(rel, last_pos, last_Me):
        """Inverse: offset + last_pos. Không cumsum."""
        d    = rel.permute(1, 0, 2)
        traj = last_pos.unsqueeze(0) + d[:, :, :2]
        me   = last_Me.unsqueeze(0)  + d[:, :, 2:]
        return traj, me

    # ── FIX 2a: Overall direction ─────────────────────────────────────────────
    def _overall_dir_loss(self, pred_abs, gt_abs, last_pos):
        """Hướng tổng thể pred vs gt từ last_pos."""
        ref = last_pos.unsqueeze(0)
        return torch.clamp(
            0.7 - (F.normalize(gt_abs - ref, p=2, dim=-1) *
                   F.normalize(pred_abs - ref, p=2, dim=-1)).sum(-1),
            min=0
        ).mean()

    # ── FIX 2b: Step-wise direction với norm guard ────────────────────────────
    def _step_dir_loss(self, pred_abs, gt_abs):
        """
        Cosine similarity giữa pred_velocity và gt_velocity từng bước.
        Norm guard: clamp > 1e-3 → gradient ổn định, không collapse.
        Mask: chỉ tính khi gt di chuyển > 0.02 (≈11 km).
        """
        if pred_abs.shape[0] < 2:
            return pred_abs.new_zeros(1).squeeze()

        pred_v = pred_abs[1:] - pred_abs[:-1]
        gt_v   = gt_abs[1:]   - gt_abs[:-1]

        pred_norm = pred_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        gt_norm   = gt_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        cos_sim   = ((pred_v / pred_norm) * (gt_v / gt_norm)).sum(-1)

        mask = (gt_v.norm(dim=-1) > 0.02).float()
        return (F.relu(0.5 - cos_sim) * mask).sum() / (mask.sum() + 1e-6)

    # ── FIX 3: Heading change loss (fix đường thẳng + recurvature) ───────────
    def _heading_change_loss(self, pred_abs, gt_abs):
        """
        Phạt khi pred không học được pattern đổi hướng của gt.

        signed_curvature[t] = cross(v[t+1], v[t]) / (|v[t+1]| * |v[t]|)
                            ∈ [-1, 1]  (dương = rẽ trái, âm = rẽ phải)

        Loss = MSE(pred_curv, gt_curv)           ← học magnitude đổi hướng
             + relu(-(pred_curv * gt_curv)).mean() ← phạt đổi hướng ngược chiều

        Tại sao tốt hơn curv_loss cũ:
          - curv_loss cũ: relu(-(pred_curl * gt_curl)) chỉ phạt dấu sai
          - heading_change: thêm MSE → học được cả magnitude của recurvature
        """
        if pred_abs.shape[0] < 3:
            return pred_abs.new_zeros(1).squeeze()

        pred_v = pred_abs[1:] - pred_abs[:-1]
        gt_v   = gt_abs[1:]   - gt_abs[:-1]

        def signed_curv(v):
            cross = v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0]
            n1 = v[1:].norm(dim=-1).clamp(min=1e-3)
            n2 = v[:-1].norm(dim=-1).clamp(min=1e-3)
            return cross / (n1 * n2)

        pred_curv = signed_curv(pred_v)
        gt_curv   = signed_curv(gt_v)

        mse_l  = F.mse_loss(pred_curv, gt_curv)
        sign_l = F.relu(-(pred_curv * gt_curv)).mean()
        return mse_l + sign_l

    def _smooth_loss(self, traj_abs):
        """Phạt acceleration lớn (jitter). Không phạt recurvature dần dần."""
        T = traj_abs.shape[0]
        if T < 3:
            return traj_abs.new_zeros(1).squeeze()
        v   = traj_abs[1:] - traj_abs[:-1]
        acc = v[1:] - v[:-1]
        return (acc ** 2).mean()

    def _weighted_disp_loss(self, pred_abs, gt_abs):
        T = pred_abs.shape[0]
        w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
        return (w * (pred_abs - gt_abs).abs()).mean()

    def _ns_pinn_loss(self, pred_abs):
        T = pred_abs.shape[0]
        if T < 4:
            return pred_abs.new_zeros(1).squeeze()
        v        = pred_abs[1:] - pred_abs[:-1]
        zeta     = (v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0])
        dzeta_dt = zeta[1:] - zeta[:-1]
        lat_rad  = torch.deg2rad(pred_abs[2:-1,:,1] * 50.0)
        beta_n   = 2 * OMEGA * lat_rad.cos() * DT_6H
        v_y_n    = v[1:-1,:,1]
        residual = dzeta_dt + beta_n * v_y_n
        return (residual ** 2).mean()

    def get_loss(self, batch_list):
        """
        v6 Loss breakdown:
          1.0 * fm_loss       OT-CFM flow matching
          1.5 * overall_dir   hướng tổng thể từ last_pos
          1.5 * step_dir      hướng từng bước (norm guard + mask)
          1.0 * disp_l        weighted displacement (xa hơn phạt nặng hơn)
          2.0 * heading_l     pattern đổi hướng = FIX chính cho recurvature
          0.2 * smooth_l      anti-jitter
          0.5 * pinn_l        vorticity constraint
        """
        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs     = batch_list[0]
        obs_Me  = batch_list[7]

        B      = traj_gt.shape[1]
        device = traj_gt.device
        lp, lm = obs[-1], obs_Me[-1]
        sm     = self.sigma_min

        x1 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)
        x0 = torch.randn_like(x1) * sm   # FIX 1: sm=0.01

        t     = torch.rand(B, device=device)
        t_exp = t.view(B, 1, 1)

        x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
        denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
        target_vel = (x1 - (1 - sm) * x_t) / denom

        pred_vel = self.net(x_t, t, batch_list)
        fm_loss  = F.mse_loss(pred_vel, target_vel)

        pred_x1     = x_t + denom * pred_vel
        pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

        overall_dir = self._overall_dir_loss(pred_abs, traj_gt, lp)
        step_dir    = self._step_dir_loss(pred_abs, traj_gt)
        disp_l      = self._weighted_disp_loss(pred_abs, traj_gt)
        heading_l   = self._heading_change_loss(pred_abs, traj_gt)
        smooth_l    = self._smooth_loss(pred_abs)
        pinn_l      = self._ns_pinn_loss(pred_abs)

        return (  1.0 * fm_loss
                + 1.5 * overall_dir
                + 1.5 * step_dir
                + 1.0 * disp_l
                + 2.0 * heading_l
                + 0.2 * smooth_l
                + 0.5 * pinn_l)

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=5, ddim_steps=10):
        obs_t, obs_m = batch_list[0], batch_list[7]
        lp, lm       = obs_t[-1], obs_m[-1]
        device       = lp.device
        B            = lp.shape[0]
        dt           = 1.0 / ddim_steps

        trajs = []
        for _ in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-3.0, 3.0)
            traj, _ = self.rel_to_abs(x_t, lp, lm)
            trajs.append(traj)

        final_traj = torch.stack(trajs).mean(0)

        mes = []
        for _ in range(max(1, num_ensemble // 2)):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-3.0, 3.0)
            _, me = self.rel_to_abs(x_t, lp, lm)
            mes.append(me)

        return final_traj, torch.stack(mes).mean(0)


TCDiffusion = TCFlowMatching