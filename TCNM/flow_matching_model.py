"""
TCNM/flow_matching_model.py  ── v2 Pure CFM
Thay thế Diffusion bằng Conditional Flow Matching thuần túy.

Bỏ hoàn toàn:
- DirectRegressionHead (CFM ổn định từ epoch 0, không cần anchor)
- blend_logit (không có gì để blend)
- steering correction / velocity anchoring (post-processing thủ công)

Giữ nguyên:
- Unet3D, Env_net, LSTM (context extraction)
- traj_to_rel / rel_to_abs
- dir_loss, curvature_loss, weighted_disp_loss

Flow Matching vs Diffusion:
  Training : x_t = (1-t)*x0 + t*x1,  loss = MSE(u_θ, x1-x0)  [1 forward pass]
  Sampling : Euler ODE 20 steps       [thay vì DDIM 100 steps]
  Tốc độ   : ~3x nhanh hơn train, ~5x nhanh hơn sample
  Ổn định  : loss giảm đều, không spike
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net


# ── Velocity Field Network ────────────────────────────────────────────────────
class VelocityField(nn.Module):
    """
    u_θ(x_t, t, context) → predicted velocity [B, pred_len, 4]

    Context = Unet3D(satellite) + Env_net(env) + LSTM(obs trajectory)
    Decoder = TransformerDecoder điều kiện trên context + time embedding
    """
    def __init__(self, pred_len=12, obs_len=8, ctx_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len

        # ── Context extractors (giữ nguyên từ diffusion) ──────────────────
        self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
        self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
        self.obs_lstm     = nn.LSTM(input_size=4, hidden_size=128,
                                    num_layers=3, batch_first=True, dropout=0.2)

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.ctx_fc1      = nn.Linear(16 + 64 + 128, 512)
        self.ctx_ln       = nn.LayerNorm(512)
        self.ctx_drop     = nn.Dropout(0.15)
        self.ctx_fc2      = nn.Linear(512, ctx_dim)

        # ── Time embedding: t ∈ [0, 1] ────────────────────────────────────
        self.time_fc1 = nn.Linear(128, 256)
        self.time_fc2 = nn.Linear(256, 128)

        # ── Trajectory → embedding ────────────────────────────────────────
        self.traj_embed = nn.Linear(4, 128)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

        # ── Transformer decoder ───────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128, nhead=8, dim_feedforward=512,
            dropout=0.15, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

    def _time_emb(self, t, dim=128):
        """Sinusoidal embedding, t ∈ [0,1] scale lên 1000 để match freq."""
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
        f_s = self.spatial_pool(f_s).flatten(1)           # [B, 16]
        f_e, _, _ = self.env_enc(env_data, image_obs)     # [B, 64]

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        _, (h_n, _) = self.obs_lstm(obs_in)
        f_h = h_n[-1]                                      # [B, 128]

        ctx = torch.cat([f_s, f_e, f_h], dim=-1)
        ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
        ctx = self.ctx_drop(ctx)
        return self.ctx_fc2(ctx)                           # [B, ctx_dim]

    def forward(self, x_t, t, batch_list):
        """
        x_t : [B, pred_len, 4]  — trajectory tại thời điểm t
        t   : [B]               — continuous time ∈ [0, 1]
        returns: velocity field [B, pred_len, 4]
        """
        ctx   = self._extract_context(batch_list)
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)                      # [B, 128]

        x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
        out    = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))    # [B, pred_len, 4]


# ── TCFlowMatching ────────────────────────────────────────────────────────────
class TCFlowMatching(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, num_steps=100, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len
        self.net      = VelocityField(pred_len, obs_len)

    # ── Coordinate helpers ────────────────────────────────────────────────────
    @staticmethod
    def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
        tf = torch.cat([last_pos.unsqueeze(0), traj_gt], dim=0)
        mf = torch.cat([last_Me.unsqueeze(0),  Me_gt],   dim=0)
        return torch.cat([tf[1:]-tf[:-1], mf[1:]-mf[:-1]], dim=-1).permute(1, 0, 2)

    @staticmethod
    def rel_to_abs(rel, last_pos, last_Me):
        d    = rel.permute(1, 0, 2)
        traj = last_pos.unsqueeze(0) + torch.cumsum(d[:, :, :2], dim=0)
        me   = last_Me.unsqueeze(0)  + torch.cumsum(d[:, :, 2:], dim=0)
        return traj, me

    # ── Geometry losses ───────────────────────────────────────────────────────
    def _dir_loss(self, pred, gt, last_pos):
        ref = last_pos.unsqueeze(0)
        return torch.clamp(
            0.7 - (F.normalize(gt - ref, p=2, dim=-1) *
                   F.normalize(pred - ref, p=2, dim=-1)).sum(-1),
            min=0
        ).mean()

    def _smooth_loss(self, rel):
        if rel.shape[1] < 2:
            return rel.new_zeros(1).squeeze()
        return ((rel[:, 1:, :2] - rel[:, :-1, :2]) ** 2).mean()

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

    # ── Training loss ─────────────────────────────────────────────────────────
    def get_loss(self, batch_list):
        """
        Pure Conditional Flow Matching loss.

        1. x1 = ground truth relative displacement
        2. x0 = noise ~ N(0, sigma)
        3. t  ~ Uniform(0, 1)
        4. x_t = (1-t)*x0 + t*x1
        5. target = x1 - x0
        6. loss = MSE(u_θ(x_t, t), target) + geometry losses
        """
        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs     = batch_list[0]
        obs_Me  = batch_list[7]

        B      = traj_gt.shape[1]
        device = traj_gt.device
        lp, lm = obs[-1], obs_Me[-1]

        # Ground truth
        x1 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)   # [B, T, 4]

        # Noise
        sigma = max((obs[-1] - obs[-2]).abs().mean().item() * 2.0
                    if obs.shape[0] >= 2 else 0.05, 0.05)
        x0 = torch.randn_like(x1) * sigma

        # Sample t
        t        = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)

        # Interpolate
        x_t        = (1 - t_expand) * x0 + t_expand * x1
        target_vel = x1 - x0

        # Predict
        pred_vel = self.net(x_t, t, batch_list)
        fm_loss  = F.mse_loss(pred_vel, target_vel)

        # Geometry losses trên predicted x1
        pred_x1     = x_t + (1 - t_expand) * pred_vel
        pred_abs, _ = self.rel_to_abs(pred_x1, lp, lm)

        dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
        smt_l  = self._smooth_loss(pred_x1)
        disp_l = self._weighted_disp_loss(pred_abs, traj_gt)
        curv_l = self._curvature_loss(pred_abs, traj_gt)

        return fm_loss + 2.0*dir_l + 0.5*smt_l + 1.0*disp_l + 1.5*curv_l

    # ── Sampling: Euler ODE ───────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=5, ddim_steps=20):
        """
        Euler ODE integration: t: 0 → 1
          x_{t+dt} = x_t + dt * u_θ(x_t, t)

        Ensemble 5 samples rồi average để giảm variance.
        ddim_steps dùng làm ode_steps (API compatible với train script).
        """
        obs_t, obs_m = batch_list[0], batch_list[7]
        lp, lm       = obs_t[-1], obs_m[-1]
        device       = lp.device
        B            = lp.shape[0]

        sigma = max((obs_t[-1] - obs_t[-2]).abs().mean().item() * 3.0
                    if obs_t.shape[0] >= 2 else 0.05, 0.05)
        dt = 1.0 / ddim_steps

        trajs = []
        for _ in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * sigma
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-1.5, 1.5)
            traj, _ = self.rel_to_abs(x_t, lp, lm)
            trajs.append(traj)

        final_traj = torch.stack(trajs).mean(0)

        mes = []
        for _ in range(max(1, num_ensemble // 2)):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * sigma
            for i in range(ddim_steps):
                t_b = torch.full((B,), i * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2] = x_t[:, :, :2].clamp(-1.5, 1.5)
            _, me = self.rel_to_abs(x_t, lp, lm)
            mes.append(me)

        pred_me = torch.stack(mes).mean(0)
        return final_traj, pred_me


# Alias: train script dùng TCDiffusion không cần đổi
TCDiffusion = TCFlowMatching