"""
TCNM/flow_matching_model.py  ── v5  OT-CFM + PINN + Absolute Trajectory
=========================================================================

ROOT CAUSE của straight-line prediction:
  v4 dùng cumulative displacement (cumsum) → model chỉ cần học
  "mỗi bước đi bao xa theo hướng hiện tại" → extrapolate thẳng là tối ưu.
  Với recurvature, bão đổi hướng đột ngột — displacement thay đổi hoàn toàn
  nhưng model không có signal để dự đoán điều đó.

FIXES vs v4:
──────────────────────────────────────────────────────────────────────────
FIX 1 │ traj_to_rel: cumulative displacement → absolute offset từ last_pos
       │ - v4: x1[t] = Σ displacement[0..t]  → mỗi step tích lũy sai số
       │ - v5: x1[t] = pos[t] - last_pos     → model học hình dạng toàn
       │   bộ trajectory, recurvature là 1 pattern shape có thể học được
       │ - rel_to_abs: bỏ cumsum, chỉ cộng offset vào last_pos
──────────────────────────────────────────────────────────────────────────
FIX 2 │ _dir_loss: thêm step-wise direction loss
       │ - v4: chỉ so hướng tổng thể pred vs gt từ last_pos
       │   → không phạt khi đúng hướng đầu nhưng sai hướng sau recurvature
       │ - v5: so cosine(pred_velocity[t], gt_velocity[t]) từng bước
       │   → phạt nặng khi hướng di chuyển từng bước sai > 60°
──────────────────────────────────────────────────────────────────────────
FIX 3 │ _smooth_loss: tính trên velocity/acceleration của abs trajectory
       │ - v4: tính trên rel tensor [B,T,4] → đo smoothness của displacement
       │ - v5: tính acceleration = Δvelocity → phạt đổi hướng ĐỘT NGỘT
       │   (không phạt recurvature dần dần, chỉ phạt noise/jitter)
──────────────────────────────────────────────────────────────────────────

Giữ nguyên: VelocityField, OT-CFM training, PINN vorticity, sampling
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


# ── Velocity Field Network ────────────────────────────────────────────────────
class VelocityField(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, ctx_dim=128, sigma_min=0.001):
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


# ── TCFlowMatching ────────────────────────────────────────────────────────────
class TCFlowMatching(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, num_steps=100,
                 sigma_min=0.001, **kwargs):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min
        self.net       = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

    # ── FIX 1: Absolute offset thay vì cumulative displacement ────────────────
    @staticmethod
    def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
        """
        v5: encode trajectory as absolute offset from last observed position.
        Model học hình dạng toàn bộ trajectory → học được recurvature.

        v4 cũ: cumulative displacement → model extrapolate thẳng
        v5 mới: offset từ last_pos → model học absolute shape
        """
        traj_norm = traj_gt - last_pos.unsqueeze(0)   # [T, B, 2]
        me_norm   = Me_gt   - last_Me.unsqueeze(0)    # [T, B, 2]
        return torch.cat([traj_norm, me_norm], dim=-1).permute(1, 0, 2)  # [B, T, 4]

    @staticmethod
    def rel_to_abs(rel, last_pos, last_Me):
        """Inverse of traj_to_rel: offset + last_pos → absolute position."""
        d    = rel.permute(1, 0, 2)            # [T, B, 4]
        traj = last_pos.unsqueeze(0) + d[:, :, :2]   # không cumsum
        me   = last_Me.unsqueeze(0)  + d[:, :, 2:]
        return traj, me

    # ── FIX 2: Step-wise direction loss ───────────────────────────────────────
    def _dir_loss(self, pred_abs, gt_abs, last_pos):
        """
        v5.1: overall direction loss + step-wise direction loss với norm guard.

        Bug v5.0: F.normalize(near-zero vector) → unstable gradient
                  model tìm shortcut minimize cosine sim mà không predict đúng hướng
                  → dir_loss collapse về 0 nhưng ADE không cải thiện

        Fix v5.1:
          - clamp norm trước khi normalize (tránh division by ~0)
          - mask: chỉ tính loss ở bước gt di chuyển đáng kể (norm > 0.01)
          - giảm step_dir weight 2.0 → 1.0 (tránh overfit direction)
        """
        # Overall direction (giữ từ v4)
        ref = last_pos.unsqueeze(0)
        overall = torch.clamp(
            0.7 - (F.normalize(gt_abs - ref, p=2, dim=-1) *
                   F.normalize(pred_abs - ref, p=2, dim=-1)).sum(-1),
            min=0
        ).mean()

        # Step-wise direction với norm guard
        if pred_abs.shape[0] >= 2:
            pred_v = pred_abs[1:] - pred_abs[:-1]          # [T-1, B, 2]
            gt_v   = gt_abs[1:]   - gt_abs[:-1]            # [T-1, B, 2]

            # clamp norm để tránh normalize vector gần 0 → gradient nhiễu
            pred_norm = pred_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            gt_norm   = gt_v.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            cos_sim   = ((pred_v / pred_norm) * (gt_v / gt_norm)).sum(-1)  # [T-1, B]

            # Chỉ phạt ở bước gt thực sự di chuyển đáng kể
            mask     = (gt_v.norm(dim=-1) > 0.01).float()  # [T-1, B]
            step_dir = (F.relu(0.5 - cos_sim) * mask).sum() / (mask.sum() + 1e-6)
        else:
            step_dir = pred_abs.new_zeros(1).squeeze()

        return overall + 1.0 * step_dir   # giảm từ 2.0 → 1.0

    def _smooth_loss(self, traj_abs):
        """
        v5: smooth loss trên abs trajectory [T, B, 2].
        Phạt acceleration lớn (noise/jitter), KHÔNG phạt recurvature dần dần.
        traj_abs: [T, B, 2]
        """
        T = traj_abs.shape[0]
        if T < 3:
            return traj_abs.new_zeros(1).squeeze()
        v   = traj_abs[1:] - traj_abs[:-1]   # velocity  [T-1, B, 2]
        acc = v[1:] - v[:-1]                  # accel     [T-2, B, 2]
        return (acc ** 2).mean()

    def _curvature_loss(self, pred_abs, gt_abs):
        if pred_abs.shape[0] < 3:
            return pred_abs.new_zeros(1).squeeze()
        pred_v    = pred_abs[1:] - pred_abs[:-1]
        gt_v      = gt_abs[1:]   - gt_abs[:-1]
        pred_curl = pred_v[1:,:,0]*pred_v[:-1,:,1] - pred_v[1:,:,1]*pred_v[:-1,:,0]
        gt_curl   = gt_v[1:,:,0] *gt_v[:-1,:,1]   - gt_v[1:,:,1] *gt_v[:-1,:,0]
        return (F.relu(-(pred_curl * gt_curl)).mean()
                + 0.3 * F.mse_loss(pred_curl, gt_curl))

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
        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs     = batch_list[0]
        obs_Me  = batch_list[7]

        B      = traj_gt.shape[1]
        device = traj_gt.device
        lp, lm = obs[-1], obs_Me[-1]
        sm     = self.sigma_min

        x1 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)
        x0 = torch.randn_like(x1) * sm

        t     = torch.rand(B, device=device)
        t_exp = t.view(B, 1, 1)

        x_t        = t_exp * x1 + (1 - t_exp * (1 - sm)) * x0
        denom      = (1 - (1 - sm) * t_exp).clamp(min=1e-5)
        target_vel = (x1 - (1 - sm) * x_t) / denom

        pred_vel = self.net(x_t, t, batch_list)
        fm_loss  = F.mse_loss(pred_vel, target_vel)

        pred_x1     = x_t + denom * pred_vel
        pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

        # v5: dir_loss bao gồm cả step-wise direction
        dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
        smt_l  = self._smooth_loss(pred_abs)        # smooth trên abs trajectory
        disp_l = self._weighted_disp_loss(pred_abs, traj_gt)
        curv_l = self._curvature_loss(pred_abs, traj_gt)
        pinn_l = self._ns_pinn_loss(pred_abs)

        return (fm_loss
                + 2.0 * dir_l     # dir_l đã bao gồm step_dir × 2.0
                + 0.3 * smt_l
                + 1.0 * disp_l
                + 1.5 * curv_l
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
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-2.0, 2.0)
            traj, _ = self.rel_to_abs(x_t, lp, lm)
            trajs.append(traj)

        final_traj = torch.stack(trajs).mean(0)

        mes = []
        for _ in range(max(1, num_ensemble // 2)):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-2.0, 2.0)
            _, me = self.rel_to_abs(x_t, lp, lm)
            mes.append(me)

        return final_traj, torch.stack(mes).mean(0)


TCDiffusion = TCFlowMatching