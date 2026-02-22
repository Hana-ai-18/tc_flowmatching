# # """
# # TCNM/diffusion_model.py  ── v5
# # FIXES so với v4:
# # 1. blend_logit có gradient trong get_loss() (trước chỉ dùng trong no_grad sample)
# # 2. Clamp displacement nới lỏng hơn (0.5 thay vì 0.3) để không kẹt khi bão di chuyển nhanh
# # 3. DDIM sampling ổn định hơn (không re-sample Me riêng, dùng chung rel)
# # 4. Regression head weight tăng early training, giảm dần khi diffusion ổn định
# # """
# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from TCNM.Unet3D_merge_tiny import Unet3D
# # from TCNM.env_net_transformer_gphsplit import Env_net


# # # ── Direct Regression Head ────────────────────────────────────────────────────
# # class DirectRegressionHead(nn.Module):
# #     """Dự báo displacement trực tiếp, ổn định từ epoch đầu."""
# #     def __init__(self, ctx_dim=128, pred_len=12):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.vel_encoder = nn.GRU(
# #             input_size=4, hidden_size=256, num_layers=2,
# #             batch_first=True, dropout=0.1
# #         )
# #         self.step_rnn  = nn.GRUCell(input_size=ctx_dim + 4, hidden_size=256)
# #         self.step_proj = nn.Sequential(
# #             nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4)
# #         )
# #         self.ctx_proj = nn.Linear(ctx_dim, ctx_dim)

# #     def forward(self, ctx, obs_traj, obs_Me):
# #         B      = ctx.shape[0]
# #         device = ctx.device

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         _, h   = self.vel_encoder(obs_in)
# #         h      = h[-1]   # [B, 256]

# #         if obs_traj.shape[0] >= 2:
# #             last_d = torch.cat([obs_traj[-1]-obs_traj[-2],
# #                                  obs_Me[-1]-obs_Me[-2]], dim=-1)
# #         else:
# #             last_d = torch.zeros(B, 4, device=device)

# #         ctx_p = self.ctx_proj(ctx)
# #         preds = []
# #         for _ in range(self.pred_len):
# #             inp   = torch.cat([ctx_p, last_d], dim=-1)
# #             h     = self.step_rnn(inp, h)
# #             delta = self.step_proj(h)
# #             preds.append(delta)
# #             last_d = delta

# #         return torch.stack(preds, dim=1)   # [B, pred_len, 4]


# # # ── Denoiser ──────────────────────────────────────────────────────────────────
# # class DeterministicDenoiser(nn.Module):
# #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=128):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.obs_len  = obs_len
# #         self.ctx_dim  = ctx_dim

# #         self.spatial_enc = Unet3D(in_channel=1, out_channel=1)
# #         self.env_enc     = Env_net(obs_len=obs_len, d_model=64)
# #         self.obs_lstm    = nn.LSTM(input_size=4, hidden_size=128,
# #                                    num_layers=3, batch_first=True, dropout=0.2)

# #         self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
# #         self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
# #         self.ctx_ln   = nn.LayerNorm(512)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# #         self.time_fc1   = nn.Linear(128, 256)
# #         self.time_fc2   = nn.Linear(256, 128)
# #         self.step_embed = nn.Linear(4, 128)

# #         decoder_layer = nn.TransformerDecoderLayer(
# #             d_model=128, nhead=8, dim_feedforward=512,
# #             dropout=0.15, activation='gelu', batch_first=True
# #         )
# #         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
# #         self.out_fc1 = nn.Linear(128, 256)
# #         self.out_fc2 = nn.Linear(256, 4)

# #         # pos_enc: detect_pred_len() đọc key này
# #         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

# #         self.reg_head   = DirectRegressionHead(ctx_dim=ctx_dim, pred_len=pred_len)

# #         # FIX: blend_logit phải tham gia vào loss để có gradient
# #         # Khởi đầu = 0 → sigmoid(0) = 0.5
# #         self.blend_logit = nn.Parameter(torch.zeros(1))

# #     def timestep_embedding(self, t, dim=128):
# #         device   = t.device
# #         half     = dim // 2
# #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=device)
# #                          * (-math.log(10000.0) / (half - 1)))
# #         emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
# #         emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         return F.pad(emb, (0, 1)) if dim % 2 else emb

# #     def _extract_context(self, batch_list):
# #         obs_traj  = batch_list[0]
# #         obs_Me    = batch_list[7]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13]

# #         f_s = self.spatial_enc(image_obs).mean(dim=2)
# #         f_s = self.spatial_pool(f_s).flatten(1)          # [B,16]

# #         f_e, _, _ = self.env_enc(env_data, image_obs)    # [B,64]

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         _, (h_n, _) = self.obs_lstm(obs_in)
# #         f_h = h_n[-1]                                     # [B,128]

# #         ctx = torch.cat([f_s, f_e, f_h], dim=-1)
# #         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
# #         ctx = self.ctx_drop(ctx)
# #         return self.ctx_fc2(ctx), obs_traj, obs_Me        # [B,128]

# #     def forward(self, x_t, t, batch_list):
# #         """Predict noise [B, pred_len, 4]"""
# #         device = x_t.device
# #         t = t.to(device)
# #         if t.dim() == 2:
# #             t = t.squeeze(1)

# #         ctx, obs_traj, obs_Me = self._extract_context(batch_list)

# #         t_emb = F.gelu(self.time_fc1(self.timestep_embedding(t)))
# #         t_emb = self.time_fc2(t_emb)

# #         x_emb  = self.step_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
# #         out    = self.transformer(x_emb, memory)
# #         return self.out_fc2(F.gelu(self.out_fc1(out)))

# #     def predict_direct(self, batch_list):
# #         ctx, obs_traj, obs_Me = self._extract_context(batch_list)
# #         return self.reg_head(ctx, obs_traj, obs_Me), ctx, obs_traj, obs_Me


# # # ── TCDiffusion ───────────────────────────────────────────────────────────────
# # class TCDiffusion(nn.Module):
# #     def __init__(self, pred_len=12, obs_len=8, num_steps=100, **kwargs):
# #         super().__init__()
# #         self.num_steps = num_steps
# #         self.pred_len  = pred_len
# #         self.obs_len   = obs_len
# #         self.denoiser  = DeterministicDenoiser(pred_len, obs_len)

# #     # ── Noise schedule ────────────────────────────────────────────────────────
# #     def _cosine_beta_schedule(self, T, s=0.008, device='cpu'):
# #         x  = torch.linspace(0, T, T + 1, device=device)
# #         ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
# #         ac = ac / ac[0]
# #         return torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.02)

# #     def _get_params(self, device):
# #         b  = self._cosine_beta_schedule(self.num_steps, device=device)
# #         a  = 1 - b
# #         ac = torch.cumprod(a, dim=0)
# #         return {'betas': b, 'alphas': a, 'ac': ac,
# #                 'sqrt_ac': ac.sqrt(), 'sqrt_1mac': (1 - ac).sqrt()}

# #     # ── Relative helpers ──────────────────────────────────────────────────────
# #     @staticmethod
# #     def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
# #         """Absolute → relative cumulative displacement [B, T, 4]"""
# #         tf = torch.cat([last_pos.unsqueeze(0), traj_gt], dim=0)
# #         mf = torch.cat([last_Me.unsqueeze(0),  Me_gt],   dim=0)
# #         return torch.cat([tf[1:]-tf[:-1], mf[1:]-mf[:-1]], dim=-1).permute(1,0,2)

# #     @staticmethod
# #     def rel_to_abs(rel, last_pos, last_Me):
# #         """Relative [B,T,4] → absolute [T,B,2], [T,B,2]"""
# #         d    = rel.permute(1, 0, 2)
# #         traj = last_pos.unsqueeze(0) + torch.cumsum(d[:,:,:2], dim=0)
# #         me   = last_Me.unsqueeze(0)  + torch.cumsum(d[:,:,2:], dim=0)
# #         return traj, me

# #     # ── Losses ────────────────────────────────────────────────────────────────
# #     def _dir_loss(self, pred, gt, last_pos):
# #         ref = last_pos.unsqueeze(0)
# #         return torch.clamp(0.7 - (F.normalize(gt-ref, p=2, dim=-1) *
# #                                    F.normalize(pred-ref, p=2, dim=-1)).sum(-1),
# #                            min=0).mean()

# #     def _smooth_loss(self, rel):
# #         if rel.shape[1] < 2: return rel.new_zeros(1).squeeze()
# #         return ((rel[:,1:,:2] - rel[:,:-1,:2])**2).mean()

# #     # ── Training loss ─────────────────────────────────────────────────────────
# #     def get_loss(self, batch_list):
# #         traj_gt = batch_list[1]   # [T, B, 2]
# #         Me_gt   = batch_list[8]
# #         obs     = batch_list[0]
# #         obs_Me  = batch_list[7]

# #         B      = traj_gt.shape[1]
# #         device = traj_gt.device
# #         lp, lm = obs[-1], obs_Me[-1]

# #         x_0 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)   # [B,T,4]

# #         p   = self._get_params(device)
# #         t   = torch.randint(0, self.num_steps, (B,), device=device).long()
# #         eps = torch.randn_like(x_0) * 0.1
# #         sac   = p['sqrt_ac'][t].view(B,1,1)
# #         s1mac = p['sqrt_1mac'][t].view(B,1,1)
# #         x_t   = sac * x_0 + s1mac * eps

# #         pred_eps = self.denoiser(x_t, t, batch_list)
# #         mse      = F.mse_loss(pred_eps, eps)

# #         pred_x0 = (x_t - s1mac * pred_eps) / (sac + 1e-8)
# #         pred_abs, _ = self.rel_to_abs(pred_x0, lp, lm)
# #         dir_l   = self._dir_loss(pred_abs, traj_gt, lp)
# #         smt_l   = self._smooth_loss(pred_x0)
# #         disp_l  = F.l1_loss(pred_abs, traj_gt)
# #         diff_l  = mse + 2.0*dir_l + 0.5*smt_l + 0.5*disp_l

# #         # ── Regression head ───────────────────────────────────────────────
# #         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
# #         reg_abs, _       = self.rel_to_abs(reg_rel, lp, lm)
# #         reg_l = (F.mse_loss(reg_abs, traj_gt)
# #                  + self._dir_loss(reg_abs, traj_gt, lp)
# #                  + self._smooth_loss(reg_rel))

# #         # ── FIX: Blend loss có gradient ───────────────────────────────────
# #         # Học blend_w qua loss: blend_w ~ 0 → tin regression hơn (ổn định)
# #         # blend_w ~ 1 → tin diffusion hơn (chính xác hơn khi đã converge)
# #         blend_w = torch.sigmoid(self.denoiser.blend_logit)          # scalar tensor
# #         blended_abs = (1 - blend_w) * reg_abs + blend_w * pred_abs
# #         blend_l = F.mse_loss(blended_abs, traj_gt)                  # có gradient!

# #         # Regularize blend: không cho = 0 hoặc = 1 cứng
# #         blend_reg = ((blend_w - 0.5)**2) * 0.01

# #         return diff_l + 0.5*reg_l + 0.3*blend_l + blend_reg

# #     # ── DDIM Sampling ─────────────────────────────────────────────────────────
# #     @torch.no_grad()
# #     def _ddim_once(self, batch_list, ddim_steps=20):
# #         B      = batch_list[0].shape[1]
# #         device = batch_list[0].device
# #         p      = self._get_params(device)

# #         obs = batch_list[0]
# #         vel = ((obs[-1] - obs[-2]).abs().mean().item()
# #                if obs.shape[0] >= 2 else 0.02)
# #         scale = max(vel * 0.5, 0.01)

# #         x_t = torch.randn(B, self.pred_len, 4, device=device) * scale

# #         steps = torch.linspace(self.num_steps-1, 0, ddim_steps,
# #                                dtype=torch.long, device=device)

# #         for i, step in enumerate(steps):
# #             sv      = step.item()
# #             t_b     = torch.full((B,), sv, device=device, dtype=torch.long)
# #             eps     = self.denoiser(x_t, t_b, batch_list)
# #             sac     = p['sqrt_ac'][sv]
# #             s1mac   = p['sqrt_1mac'][sv]
# #             x0      = ((x_t - s1mac * eps) / (sac + 1e-8)).clamp(-0.5, 0.5)

# #             if i < len(steps) - 1:
# #                 ns   = steps[i+1].item()
# #                 x_t  = p['sqrt_ac'][ns] * x0 + p['sqrt_1mac'][ns] * eps
# #             else:
# #                 x_t = x0

# #             # FIX: clamp nới lỏng hơn (0.5 thay vì 0.3)
# #             # 0.5 units ≈ 0.5×50×0.1°×111km = ~278km/6h (vẫn hợp lý cho bão mạnh)
# #             x_t[:,:,:2] = x_t[:,:,:2].clamp(-0.5, 0.5)

# #         return x_t   # [B, pred_len, 4]

# #     @torch.no_grad()
# #     def sample(self, batch_list, num_ensemble=5, ddim_steps=20):
# #         """
# #         Ensemble DDIM + Regression blend.
# #         blend_w được học tự động qua get_loss().
# #         """
# #         obs_t, obs_m = batch_list[0], batch_list[7]
# #         lp, lm       = obs_t[-1], obs_m[-1]

# #         # DDIM ensemble
# #         trajs = []
# #         for _ in range(num_ensemble):
# #             rel = self._ddim_once(batch_list, ddim_steps)
# #             t, _ = self.rel_to_abs(rel, lp, lm)
# #             trajs.append(t)
# #         ens_traj = torch.stack(trajs).mean(0)   # [T, B, 2]

# #         # Regression
# #         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
# #         reg_traj, reg_me = self.rel_to_abs(reg_rel, lp, lm)

# #         # Blend
# #         bw          = torch.sigmoid(self.denoiser.blend_logit).item()
# #         final_traj  = (1 - bw) * reg_traj + bw * ens_traj

# #         # Me: từ ensemble
# #         mes = []
# #         for _ in range(max(1, num_ensemble//2)):
# #             rel = self._ddim_once(batch_list, ddim_steps)
# #             _, m = self.rel_to_abs(rel, lp, lm)
# #             mes.append(m)
# #         pred_me = torch.stack(mes).mean(0)

# #         return final_traj, pred_me


# # _______________________________ 362km_________________________
# # """
# # TCNM/diffusion_model.py  ── v5
# # FIXES so với v4:
# # 1. blend_logit có gradient trong get_loss() (trước chỉ dùng trong no_grad sample)
# # 2. Clamp displacement nới lỏng hơn (0.5 thay vì 0.3) để không kẹt khi bão di chuyển nhanh
# # 3. DDIM sampling ổn định hơn (không re-sample Me riêng, dùng chung rel)
# # 4. Regression head weight tăng early training, giảm dần khi diffusion ổn định
# # """
# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from TCNM.Unet3D_merge_tiny import Unet3D
# # from TCNM.env_net_transformer_gphsplit import Env_net


# # # ── Direct Regression Head ────────────────────────────────────────────────────
# # class DirectRegressionHead(nn.Module):
# #     """Dự báo displacement trực tiếp, ổn định từ epoch đầu."""
# #     def __init__(self, ctx_dim=128, pred_len=12):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.vel_encoder = nn.GRU(
# #             input_size=4, hidden_size=256, num_layers=2,
# #             batch_first=True, dropout=0.1
# #         )
# #         self.step_rnn  = nn.GRUCell(input_size=ctx_dim + 4, hidden_size=256)
# #         self.step_proj = nn.Sequential(
# #             nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4)
# #         )
# #         self.ctx_proj = nn.Linear(ctx_dim, ctx_dim)

# #     def forward(self, ctx, obs_traj, obs_Me):
# #         B      = ctx.shape[0]
# #         device = ctx.device

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         _, h   = self.vel_encoder(obs_in)
# #         h      = h[-1]   # [B, 256]

# #         if obs_traj.shape[0] >= 2:
# #             last_d = torch.cat([obs_traj[-1]-obs_traj[-2],
# #                                  obs_Me[-1]-obs_Me[-2]], dim=-1)
# #         else:
# #             last_d = torch.zeros(B, 4, device=device)

# #         ctx_p = self.ctx_proj(ctx)
# #         preds = []
# #         for _ in range(self.pred_len):
# #             inp   = torch.cat([ctx_p, last_d], dim=-1)
# #             h     = self.step_rnn(inp, h)
# #             delta = self.step_proj(h)
# #             preds.append(delta)
# #             last_d = delta

# #         return torch.stack(preds, dim=1)   # [B, pred_len, 4]


# # # ── Denoiser ──────────────────────────────────────────────────────────────────
# # class DeterministicDenoiser(nn.Module):
# #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=128):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.obs_len  = obs_len
# #         self.ctx_dim  = ctx_dim

# #         self.spatial_enc = Unet3D(in_channel=1, out_channel=1)
# #         self.env_enc     = Env_net(obs_len=obs_len, d_model=64)
# #         self.obs_lstm    = nn.LSTM(input_size=4, hidden_size=128,
# #                                    num_layers=3, batch_first=True, dropout=0.2)

# #         self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
# #         self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
# #         self.ctx_ln   = nn.LayerNorm(512)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# #         self.time_fc1   = nn.Linear(128, 256)
# #         self.time_fc2   = nn.Linear(256, 128)
# #         self.step_embed = nn.Linear(4, 128)

# #         decoder_layer = nn.TransformerDecoderLayer(
# #             d_model=128, nhead=8, dim_feedforward=512,
# #             dropout=0.15, activation='gelu', batch_first=True
# #         )
# #         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
# #         self.out_fc1 = nn.Linear(128, 256)
# #         self.out_fc2 = nn.Linear(256, 4)

# #         # pos_enc: detect_pred_len() đọc key này
# #         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

# #         self.reg_head   = DirectRegressionHead(ctx_dim=ctx_dim, pred_len=pred_len)

# #         # FIX: blend_logit phải tham gia vào loss để có gradient
# #         # Khởi đầu = 0 → sigmoid(0) = 0.5
# #         self.blend_logit = nn.Parameter(torch.zeros(1))

# #     def timestep_embedding(self, t, dim=128):
# #         device   = t.device
# #         half     = dim // 2
# #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=device)
# #                          * (-math.log(10000.0) / (half - 1)))
# #         emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
# #         emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         return F.pad(emb, (0, 1)) if dim % 2 else emb

# #     def _extract_context(self, batch_list):
# #         obs_traj  = batch_list[0]
# #         obs_Me    = batch_list[7]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13]

# #         f_s = self.spatial_enc(image_obs).mean(dim=2)
# #         f_s = self.spatial_pool(f_s).flatten(1)          # [B,16]

# #         f_e, _, _ = self.env_enc(env_data, image_obs)    # [B,64]

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         _, (h_n, _) = self.obs_lstm(obs_in)
# #         f_h = h_n[-1]                                     # [B,128]

# #         ctx = torch.cat([f_s, f_e, f_h], dim=-1)
# #         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
# #         ctx = self.ctx_drop(ctx)
# #         return self.ctx_fc2(ctx), obs_traj, obs_Me        # [B,128]

# #     def forward(self, x_t, t, batch_list):
# #         """Predict noise [B, pred_len, 4]"""
# #         device = x_t.device
# #         t = t.to(device)
# #         if t.dim() == 2:
# #             t = t.squeeze(1)

# #         ctx, obs_traj, obs_Me = self._extract_context(batch_list)

# #         t_emb = F.gelu(self.time_fc1(self.timestep_embedding(t)))
# #         t_emb = self.time_fc2(t_emb)

# #         x_emb  = self.step_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
# #         out    = self.transformer(x_emb, memory)
# #         return self.out_fc2(F.gelu(self.out_fc1(out)))

# #     def predict_direct(self, batch_list):
# #         ctx, obs_traj, obs_Me = self._extract_context(batch_list)
# #         return self.reg_head(ctx, obs_traj, obs_Me), ctx, obs_traj, obs_Me


# # # ── TCDiffusion ───────────────────────────────────────────────────────────────
# # class TCDiffusion(nn.Module):
# #     def __init__(self, pred_len=12, obs_len=8, num_steps=100, **kwargs):
# #         super().__init__()
# #         self.num_steps = num_steps
# #         self.pred_len  = pred_len
# #         self.obs_len   = obs_len
# #         self.denoiser  = DeterministicDenoiser(pred_len, obs_len)

# #     # ── Noise schedule ────────────────────────────────────────────────────────
# #     def _cosine_beta_schedule(self, T, s=0.008, device='cpu'):
# #         x  = torch.linspace(0, T, T + 1, device=device)
# #         ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
# #         ac = ac / ac[0]
# #         return torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.02)

# #     def _get_params(self, device):
# #         b  = self._cosine_beta_schedule(self.num_steps, device=device)
# #         a  = 1 - b
# #         ac = torch.cumprod(a, dim=0)
# #         return {'betas': b, 'alphas': a, 'ac': ac,
# #                 'sqrt_ac': ac.sqrt(), 'sqrt_1mac': (1 - ac).sqrt()}

# #     # ── Relative helpers ──────────────────────────────────────────────────────
# #     @staticmethod
# #     def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
# #         """Absolute → relative cumulative displacement [B, T, 4]"""
# #         tf = torch.cat([last_pos.unsqueeze(0), traj_gt], dim=0)
# #         mf = torch.cat([last_Me.unsqueeze(0),  Me_gt],   dim=0)
# #         return torch.cat([tf[1:]-tf[:-1], mf[1:]-mf[:-1]], dim=-1).permute(1,0,2)

# #     @staticmethod
# #     def rel_to_abs(rel, last_pos, last_Me):
# #         """Relative [B,T,4] → absolute [T,B,2], [T,B,2]"""
# #         d    = rel.permute(1, 0, 2)
# #         traj = last_pos.unsqueeze(0) + torch.cumsum(d[:,:,:2], dim=0)
# #         me   = last_Me.unsqueeze(0)  + torch.cumsum(d[:,:,2:], dim=0)
# #         return traj, me

# #     # ── Losses ────────────────────────────────────────────────────────────────
# #     def _dir_loss(self, pred, gt, last_pos):
# #         ref = last_pos.unsqueeze(0)
# #         return torch.clamp(0.7 - (F.normalize(gt-ref, p=2, dim=-1) *
# #                                    F.normalize(pred-ref, p=2, dim=-1)).sum(-1),
# #                            min=0).mean()

# #     def _smooth_loss(self, rel):
# #         if rel.shape[1] < 2: return rel.new_zeros(1).squeeze()
# #         return ((rel[:,1:,:2] - rel[:,:-1,:2])**2).mean()

# #     def _curvature_loss(self, pred_abs, gt_abs):
# #         """
# #         Penalize khi track dự đoán không uốn cong giống ground truth.
# #         Tính bằng cross-product của velocity vectors liên tiếp.
# #         pred_abs / gt_abs: [T, B, 2]
# #         """
# #         if pred_abs.shape[0] < 3:
# #             return pred_abs.new_zeros(1).squeeze()

# #         # Velocity vectors
# #         pred_v = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
# #         gt_v   = gt_abs[1:]   - gt_abs[:-1]

# #         # Angular change (cross product z-component: vx*wy - vy*wx)
# #         pred_curl = pred_v[1:,:,0]*pred_v[:-1,:,1] - pred_v[1:,:,1]*pred_v[:-1,:,0]
# #         gt_curl   = gt_v[1:,:,0]  *gt_v[:-1,:,1]   - gt_v[1:,:,1]  *gt_v[:-1,:,0]

# #         # Penalize sign mismatch (uốn ngược chiều) và magnitude mismatch
# #         sign_loss = F.relu(-(pred_curl * gt_curl)).mean()           # wrong turn direction
# #         mag_loss  = F.mse_loss(pred_curl, gt_curl)                  # wrong curvature amount
# #         return sign_loss + 0.3 * mag_loss

# #     def _weighted_disp_loss(self, pred_abs, gt_abs):
# #         """
# #         L1 loss với weight tăng dần theo thời gian.
# #         Bước xa (48h, 72h) bị penalize nặng hơn → model học uốn cong đúng ở cuối.
# #         pred_abs / gt_abs: [T, B, 2]
# #         """
# #         T = pred_abs.shape[0]
# #         # Weight: từ 1.0 (bước đầu) → 2.5 (bước cuối)
# #         w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
# #         return (w * (pred_abs - gt_abs).abs()).mean()

# #     # ── Training loss ─────────────────────────────────────────────────────────
# #     def get_loss(self, batch_list):
# #         traj_gt = batch_list[1]   # [T, B, 2]
# #         Me_gt   = batch_list[8]
# #         obs     = batch_list[0]
# #         obs_Me  = batch_list[7]

# #         B      = traj_gt.shape[1]
# #         device = traj_gt.device
# #         lp, lm = obs[-1], obs_Me[-1]

# #         x_0 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)   # [B,T,4]

# #         p   = self._get_params(device)
# #         t   = torch.randint(0, self.num_steps, (B,), device=device).long()
# #         eps = torch.randn_like(x_0) * 0.1
# #         sac   = p['sqrt_ac'][t].view(B,1,1)
# #         s1mac = p['sqrt_1mac'][t].view(B,1,1)
# #         x_t   = sac * x_0 + s1mac * eps

# #         pred_eps = self.denoiser(x_t, t, batch_list)
# #         mse      = F.mse_loss(pred_eps, eps)

# #         pred_x0 = (x_t - s1mac * pred_eps) / (sac + 1e-8)
# #         pred_abs, _ = self.rel_to_abs(pred_x0, lp, lm)

# #         dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
# #         smt_l  = self._smooth_loss(pred_x0)
# #         disp_l = self._weighted_disp_loss(pred_abs, traj_gt)  # weighted L1
# #         curv_l = self._curvature_loss(pred_abs, traj_gt)       # curvature

# #         diff_l = mse + 2.0*dir_l + 0.5*smt_l + 1.0*disp_l + 1.5*curv_l

# #         # ── Regression head ───────────────────────────────────────────────
# #         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
# #         reg_abs, _       = self.rel_to_abs(reg_rel, lp, lm)
# #         reg_l = (self._weighted_disp_loss(reg_abs, traj_gt)
# #                  + self._dir_loss(reg_abs, traj_gt, lp)
# #                  + self._smooth_loss(reg_rel)
# #                  + self._curvature_loss(reg_abs, traj_gt))

# #         # ── Blend loss có gradient ────────────────────────────────────────
# #         blend_w = torch.sigmoid(self.denoiser.blend_logit)
# #         blended_abs = (1 - blend_w) * reg_abs + blend_w * pred_abs
# #         blend_l = self._weighted_disp_loss(blended_abs, traj_gt)

# #         blend_reg = ((blend_w - 0.5)**2) * 0.01

# #         return diff_l + 0.5*reg_l + 0.3*blend_l + blend_reg

# #     # ── DDIM Sampling ─────────────────────────────────────────────────────────
# #     @torch.no_grad()
# #     def _ddim_once(self, batch_list, ddim_steps=20):
# #         B      = batch_list[0].shape[1]
# #         device = batch_list[0].device
# #         p      = self._get_params(device)

# #         obs = batch_list[0]
# #         vel = ((obs[-1] - obs[-2]).abs().mean().item()
# #                if obs.shape[0] >= 2 else 0.02)
# #         scale = max(vel * 0.5, 0.01)

# #         x_t = torch.randn(B, self.pred_len, 4, device=device) * scale

# #         steps = torch.linspace(self.num_steps-1, 0, ddim_steps,
# #                                dtype=torch.long, device=device)

# #         for i, step in enumerate(steps):
# #             sv      = step.item()
# #             t_b     = torch.full((B,), sv, device=device, dtype=torch.long)
# #             eps     = self.denoiser(x_t, t_b, batch_list)
# #             sac     = p['sqrt_ac'][sv]
# #             s1mac   = p['sqrt_1mac'][sv]
# #             x0      = ((x_t - s1mac * eps) / (sac + 1e-8)).clamp(-0.5, 0.5)

# #             if i < len(steps) - 1:
# #                 ns   = steps[i+1].item()
# #                 x_t  = p['sqrt_ac'][ns] * x0 + p['sqrt_1mac'][ns] * eps
# #             else:
# #                 x_t = x0

# #             # FIX: clamp nới lỏng hơn (0.5 thay vì 0.3)
# #             # 0.5 units ≈ 0.5×50×0.1°×111km = ~278km/6h (vẫn hợp lý cho bão mạnh)
# #             x_t[:,:,:2] = x_t[:,:,:2].clamp(-0.5, 0.5)

# #         return x_t   # [B, pred_len, 4]

# #     @torch.no_grad()
# #     def sample(self, batch_list, num_ensemble=5, ddim_steps=20):
# #         """
# #         Ensemble DDIM + Regression blend + Velocity bias correction.

# #         Bias correction:
# #         - Nhìn vào lịch sử quan sát: bão đang đi theo hướng nào?
# #         - Nếu predicted mean velocity lệch nhiều so với observed velocity,
# #           kéo nhẹ về phía observed direction (alpha=0.3)
# #         - Điều này giúp giảm systematic drift (ví dụ lệch bắc như trong ảnh)
# #         """
# #         obs_t, obs_m = batch_list[0], batch_list[7]
# #         lp, lm       = obs_t[-1], obs_m[-1]

# #         # ── Tính observed velocity (mean of last 3 steps) ─────────────────
# #         n_vel = min(3, obs_t.shape[0] - 1)
# #         obs_vel = (obs_t[-1] - obs_t[-1-n_vel]) / n_vel   # [B, 2] mean vel/step

# #         # ── DDIM ensemble ──────────────────────────────────────────────────
# #         all_rels = []
# #         trajs    = []
# #         for _ in range(num_ensemble):
# #             rel = self._ddim_once(batch_list, ddim_steps)
# #             all_rels.append(rel)
# #             t, _ = self.rel_to_abs(rel, lp, lm)
# #             trajs.append(t)
# #         ens_traj = torch.stack(trajs).mean(0)   # [T, B, 2]

# #         # ── Velocity bias correction ───────────────────────────────────────
# #         # Mean predicted velocity (bước đầu tiên)
# #         mean_rel = torch.stack(all_rels).mean(0)        # [B, pred_len, 4]
# #         pred_vel = mean_rel[:, 0, :2]                    # [B, 2] bước đầu

# #         # Bias = sai lệch hướng ban đầu
# #         vel_bias = pred_vel - obs_vel                    # [B, 2]

# #         # Correction: trừ đi 30% bias, áp dụng giảm dần theo thời gian
# #         T = ens_traj.shape[0]
# #         decay = torch.linspace(0.30, 0.05, T, device=ens_traj.device).view(T, 1, 1)
# #         # Tích lũy correction theo cumsum
# #         correction = vel_bias.unsqueeze(0) * decay       # [T, B, 2]
# #         correction = torch.cumsum(correction, dim=0)     # [T, B, 2]
# #         ens_traj_corrected = ens_traj - correction

# #         # ── Direct Regression ─────────────────────────────────────────────
# #         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
# #         reg_traj, reg_me = self.rel_to_abs(reg_rel, lp, lm)

# #         # Cũng apply bias correction cho regression
# #         reg_pred_vel = reg_rel[:, 0, :2]                # [B, 2]
# #         reg_bias     = reg_pred_vel - obs_vel
# #         reg_correction = reg_bias.unsqueeze(0) * decay
# #         reg_correction = torch.cumsum(reg_correction, dim=0)
# #         reg_traj_corrected = reg_traj - reg_correction

# #         # ── Blend ─────────────────────────────────────────────────────────
# #         bw         = torch.sigmoid(self.denoiser.blend_logit).item()
# #         final_traj = (1 - bw) * reg_traj_corrected + bw * ens_traj_corrected

# #         # ── pred_Me ───────────────────────────────────────────────────────
# #         mes = []
# #         for _ in range(max(1, num_ensemble//2)):
# #             rel = self._ddim_once(batch_list, ddim_steps)
# #             _, m = self.rel_to_abs(rel, lp, lm)
# #             mes.append(m)
# #         pred_me = torch.stack(mes).mean(0)

# #         return final_traj, pred_me

# # _________________new version________________________
# """
# TCNM/diffusion_model.py  ── v5
# FIXES so với v4:
# 1. blend_logit có gradient trong get_loss() (trước chỉ dùng trong no_grad sample)
# 2. Clamp displacement nới lỏng hơn (0.5 thay vì 0.3) để không kẹt khi bão di chuyển nhanh
# 3. DDIM sampling ổn định hơn (không re-sample Me riêng, dùng chung rel)
# 4. Regression head weight tăng early training, giảm dần khi diffusion ổn định
# """
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from TCNM.Unet3D_merge_tiny import Unet3D
# from TCNM.env_net_transformer_gphsplit import Env_net


# # ── Direct Regression Head ────────────────────────────────────────────────────
# class DirectRegressionHead(nn.Module):
#     """
#     Dự báo displacement có khả năng học TURNING (thay đổi hướng).

#     Key fix: input vào step_rnn bao gồm cả acceleration (delta_vel)
#     chứ không chỉ velocity → model có thể học bão đang bắt đầu recurve.
#     """
#     def __init__(self, ctx_dim=128, pred_len=12):
#         super().__init__()
#         self.pred_len = pred_len

#         # Encode obs: velocity + acceleration
#         # input = [delta_pos(2), delta_Me(2), accel(2)] = 6 dims
#         self.vel_encoder = nn.GRU(
#             input_size=6, hidden_size=256, num_layers=2,
#             batch_first=True, dropout=0.1
#         )

#         # step_rnn input: ctx + vel(2) + accel(2) + Me_vel(2) = ctx+6
#         self.step_rnn  = nn.GRUCell(input_size=ctx_dim + 6, hidden_size=256)
#         self.step_proj = nn.Sequential(
#             nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4)
#         )
#         # Predict acceleration correction (how much to turn)
#         self.accel_proj = nn.Sequential(
#             nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2)
#         )
#         self.ctx_proj = nn.Linear(ctx_dim, ctx_dim)

#     def forward(self, ctx, obs_traj, obs_Me):
#         B      = ctx.shape[0]
#         device = ctx.device

#         # Build obs sequence: [vel, Me_vel, accel]
#         if obs_traj.shape[0] >= 2:
#             vel_seq    = obs_traj[1:] - obs_traj[:-1]       # [T-1, B, 2]
#             Me_vel_seq = obs_Me[1:]   - obs_Me[:-1]
#         else:
#             vel_seq    = torch.zeros(1, B, 2, device=device)
#             Me_vel_seq = torch.zeros(1, B, 2, device=device)

#         if vel_seq.shape[0] >= 2:
#             accel_seq = vel_seq[1:] - vel_seq[:-1]          # [T-2, B, 2]
#             # pad to same length
#             accel_seq = torch.cat([accel_seq[:1], accel_seq], dim=0)
#         else:
#             accel_seq = torch.zeros_like(vel_seq)

#         # Take min length
#         T_seq = min(vel_seq.shape[0], accel_seq.shape[0])
#         obs_in = torch.cat([
#             vel_seq[-T_seq:],
#             Me_vel_seq[-T_seq:],
#             accel_seq[-T_seq:]
#         ], dim=-1).permute(1, 0, 2)   # [B, T_seq, 6]

#         _, h = self.vel_encoder(obs_in)
#         h    = h[-1]   # [B, 256]

#         # Initial conditions: last velocity + last acceleration
#         last_vel   = vel_seq[-1]      # [B, 2]
#         last_Me_v  = Me_vel_seq[-1]   # [B, 2]
#         if vel_seq.shape[0] >= 2:
#             last_accel = vel_seq[-1] - vel_seq[-2]
#         else:
#             last_accel = torch.zeros(B, 2, device=device)

#         ctx_p = self.ctx_proj(ctx)
#         preds = []
#         cur_vel   = last_vel
#         cur_accel = last_accel
#         cur_Me_v  = last_Me_v

#         for _ in range(self.pred_len):
#             inp = torch.cat([ctx_p, cur_vel, cur_accel, cur_Me_v], dim=-1)
#             h   = self.step_rnn(inp, h)

#             # Predict displacement and acceleration correction
#             delta      = self.step_proj(h)         # [B, 4]
#             accel_corr = self.accel_proj(h)         # [B, 2] — how to turn

#             preds.append(delta)

#             # Update velocity with learned acceleration
#             cur_accel = accel_corr
#             cur_vel   = delta[:, :2]
#             cur_Me_v  = delta[:, 2:]

#         return torch.stack(preds, dim=1)   # [B, pred_len, 4]


# # ── Denoiser ──────────────────────────────────────────────────────────────────
# class DeterministicDenoiser(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=128):
#         super().__init__()
#         self.pred_len = pred_len
#         self.obs_len  = obs_len
#         self.ctx_dim  = ctx_dim

#         self.spatial_enc = Unet3D(in_channel=1, out_channel=1)
#         self.env_enc     = Env_net(obs_len=obs_len, d_model=64)
#         self.obs_lstm    = nn.LSTM(input_size=4, hidden_size=128,
#                                    num_layers=3, batch_first=True, dropout=0.2)

#         self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
#         self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
#         self.ctx_ln   = nn.LayerNorm(512)
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(512, ctx_dim)

#         self.time_fc1   = nn.Linear(128, 256)
#         self.time_fc2   = nn.Linear(256, 128)
#         self.step_embed = nn.Linear(4, 128)

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=128, nhead=8, dim_feedforward=512,
#             dropout=0.15, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
#         self.out_fc1 = nn.Linear(128, 256)
#         self.out_fc2 = nn.Linear(256, 4)

#         # pos_enc: detect_pred_len() đọc key này
#         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

#         self.reg_head   = DirectRegressionHead(ctx_dim=ctx_dim, pred_len=pred_len)

#         # FIX: blend_logit phải tham gia vào loss để có gradient
#         # Khởi đầu = 0 → sigmoid(0) = 0.5
#         self.blend_logit = nn.Parameter(torch.zeros(1))

#     def timestep_embedding(self, t, dim=128):
#         device   = t.device
#         half     = dim // 2
#         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=device)
#                          * (-math.log(10000.0) / (half - 1)))
#         emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
#         emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return F.pad(emb, (0, 1)) if dim % 2 else emb

#     def _extract_context(self, batch_list):
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]

#         f_s = self.spatial_enc(image_obs).mean(dim=2)
#         f_s = self.spatial_pool(f_s).flatten(1)          # [B,16]

#         f_e, _, _ = self.env_enc(env_data, image_obs)    # [B,64]

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         _, (h_n, _) = self.obs_lstm(obs_in)
#         f_h = h_n[-1]                                     # [B,128]

#         ctx = torch.cat([f_s, f_e, f_h], dim=-1)
#         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
#         ctx = self.ctx_drop(ctx)
#         return self.ctx_fc2(ctx), obs_traj, obs_Me        # [B,128]

#     def forward(self, x_t, t, batch_list):
#         """Predict noise [B, pred_len, 4]"""
#         device = x_t.device
#         t = t.to(device)
#         if t.dim() == 2:
#             t = t.squeeze(1)

#         ctx, obs_traj, obs_Me = self._extract_context(batch_list)

#         t_emb = F.gelu(self.time_fc1(self.timestep_embedding(t)))
#         t_emb = self.time_fc2(t_emb)

#         x_emb  = self.step_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
#         out    = self.transformer(x_emb, memory)
#         return self.out_fc2(F.gelu(self.out_fc1(out)))

#     def predict_direct(self, batch_list):
#         ctx, obs_traj, obs_Me = self._extract_context(batch_list)
#         return self.reg_head(ctx, obs_traj, obs_Me), ctx, obs_traj, obs_Me


# # ── TCDiffusion ───────────────────────────────────────────────────────────────
# class TCDiffusion(nn.Module):
#     def __init__(self, pred_len=12, obs_len=8, num_steps=100, **kwargs):
#         super().__init__()
#         self.num_steps = num_steps
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.denoiser  = DeterministicDenoiser(pred_len, obs_len)

#     # ── Noise schedule ────────────────────────────────────────────────────────
#     def _cosine_beta_schedule(self, T, s=0.008, device='cpu'):
#         x  = torch.linspace(0, T, T + 1, device=device)
#         ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
#         ac = ac / ac[0]
#         return torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.02)

#     def _get_params(self, device):
#         b  = self._cosine_beta_schedule(self.num_steps, device=device)
#         a  = 1 - b
#         ac = torch.cumprod(a, dim=0)
#         return {'betas': b, 'alphas': a, 'ac': ac,
#                 'sqrt_ac': ac.sqrt(), 'sqrt_1mac': (1 - ac).sqrt()}

#     # ── Relative helpers ──────────────────────────────────────────────────────
#     @staticmethod
#     def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
#         """Absolute → relative cumulative displacement [B, T, 4]"""
#         tf = torch.cat([last_pos.unsqueeze(0), traj_gt], dim=0)
#         mf = torch.cat([last_Me.unsqueeze(0),  Me_gt],   dim=0)
#         return torch.cat([tf[1:]-tf[:-1], mf[1:]-mf[:-1]], dim=-1).permute(1,0,2)

#     @staticmethod
#     def rel_to_abs(rel, last_pos, last_Me):
#         """Relative [B,T,4] → absolute [T,B,2], [T,B,2]"""
#         d    = rel.permute(1, 0, 2)
#         traj = last_pos.unsqueeze(0) + torch.cumsum(d[:,:,:2], dim=0)
#         me   = last_Me.unsqueeze(0)  + torch.cumsum(d[:,:,2:], dim=0)
#         return traj, me

#     # ── Losses ────────────────────────────────────────────────────────────────
#     def _dir_loss(self, pred, gt, last_pos):
#         ref = last_pos.unsqueeze(0)
#         return torch.clamp(0.7 - (F.normalize(gt-ref, p=2, dim=-1) *
#                                    F.normalize(pred-ref, p=2, dim=-1)).sum(-1),
#                            min=0).mean()

#     def _smooth_loss(self, rel):
#         if rel.shape[1] < 2: return rel.new_zeros(1).squeeze()
#         return ((rel[:,1:,:2] - rel[:,:-1,:2])**2).mean()

#     def _curvature_loss(self, pred_abs, gt_abs):
#         """
#         Penalize khi track dự đoán không uốn cong giống ground truth.
#         Tính bằng cross-product của velocity vectors liên tiếp.
#         pred_abs / gt_abs: [T, B, 2]
#         """
#         if pred_abs.shape[0] < 3:
#             return pred_abs.new_zeros(1).squeeze()

#         # Velocity vectors
#         pred_v = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
#         gt_v   = gt_abs[1:]   - gt_abs[:-1]

#         # Angular change (cross product z-component: vx*wy - vy*wx)
#         pred_curl = pred_v[1:,:,0]*pred_v[:-1,:,1] - pred_v[1:,:,1]*pred_v[:-1,:,0]
#         gt_curl   = gt_v[1:,:,0]  *gt_v[:-1,:,1]   - gt_v[1:,:,1]  *gt_v[:-1,:,0]

#         # Penalize sign mismatch (uốn ngược chiều) và magnitude mismatch
#         sign_loss = F.relu(-(pred_curl * gt_curl)).mean()           # wrong turn direction
#         mag_loss  = F.mse_loss(pred_curl, gt_curl)                  # wrong curvature amount
#         return sign_loss + 0.3 * mag_loss

#     def _weighted_disp_loss(self, pred_abs, gt_abs):
#         """
#         L1 loss với weight tăng dần theo thời gian.
#         Bước xa (48h, 72h) bị penalize nặng hơn → model học uốn cong đúng ở cuối.
#         pred_abs / gt_abs: [T, B, 2]
#         """
#         T = pred_abs.shape[0]
#         # Weight: từ 1.0 (bước đầu) → 2.5 (bước cuối)
#         w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
#         return (w * (pred_abs - gt_abs).abs()).mean()

#     # ── Training loss ─────────────────────────────────────────────────────────
#     def get_loss(self, batch_list):
#         traj_gt = batch_list[1]   # [T, B, 2]
#         Me_gt   = batch_list[8]
#         obs     = batch_list[0]
#         obs_Me  = batch_list[7]

#         B      = traj_gt.shape[1]
#         device = traj_gt.device
#         lp, lm = obs[-1], obs_Me[-1]

#         x_0 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)   # [B,T,4]

#         p   = self._get_params(device)
#         t   = torch.randint(0, self.num_steps, (B,), device=device).long()
#         eps = torch.randn_like(x_0) * 0.1
#         sac   = p['sqrt_ac'][t].view(B,1,1)
#         s1mac = p['sqrt_1mac'][t].view(B,1,1)
#         x_t   = sac * x_0 + s1mac * eps

#         pred_eps = self.denoiser(x_t, t, batch_list)
#         mse      = F.mse_loss(pred_eps, eps)

#         pred_x0 = (x_t - s1mac * pred_eps) / (sac + 1e-8)
#         pred_abs, _ = self.rel_to_abs(pred_x0, lp, lm)

#         dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
#         smt_l  = self._smooth_loss(pred_x0)
#         disp_l = self._weighted_disp_loss(pred_abs, traj_gt)  # weighted L1
#         curv_l = self._curvature_loss(pred_abs, traj_gt)       # curvature

#         diff_l = mse + 2.0*dir_l + 0.5*smt_l + 1.0*disp_l + 1.5*curv_l

#         # ── Regression head ───────────────────────────────────────────────
#         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
#         reg_abs, _       = self.rel_to_abs(reg_rel, lp, lm)
#         reg_l = (self._weighted_disp_loss(reg_abs, traj_gt)
#                  + self._dir_loss(reg_abs, traj_gt, lp)
#                  + self._smooth_loss(reg_rel)
#                  + self._curvature_loss(reg_abs, traj_gt))

#         # ── Blend loss có gradient ────────────────────────────────────────
#         blend_w = torch.sigmoid(self.denoiser.blend_logit)
#         blended_abs = (1 - blend_w) * reg_abs + blend_w * pred_abs
#         blend_l = self._weighted_disp_loss(blended_abs, traj_gt)

#         blend_reg = ((blend_w - 0.5)**2) * 0.01

#         return diff_l + 0.5*reg_l + 0.3*blend_l + blend_reg

#     # ── DDIM Sampling ─────────────────────────────────────────────────────────
#     @torch.no_grad()
#     def _ddim_once(self, batch_list, ddim_steps=20):
#         B      = batch_list[0].shape[1]
#         device = batch_list[0].device
#         p      = self._get_params(device)

#         obs = batch_list[0]
#         vel = ((obs[-1] - obs[-2]).abs().mean().item()
#                if obs.shape[0] >= 2 else 0.02)
#         scale = max(vel * 0.5, 0.01)

#         x_t = torch.randn(B, self.pred_len, 4, device=device) * scale

#         steps = torch.linspace(self.num_steps-1, 0, ddim_steps,
#                                dtype=torch.long, device=device)

#         for i, step in enumerate(steps):
#             sv      = step.item()
#             t_b     = torch.full((B,), sv, device=device, dtype=torch.long)
#             eps     = self.denoiser(x_t, t_b, batch_list)
#             sac     = p['sqrt_ac'][sv]
#             s1mac   = p['sqrt_1mac'][sv]
#             x0      = ((x_t - s1mac * eps) / (sac + 1e-8)).clamp(-0.5, 0.5)

#             if i < len(steps) - 1:
#                 ns   = steps[i+1].item()
#                 x_t  = p['sqrt_ac'][ns] * x0 + p['sqrt_1mac'][ns] * eps
#             else:
#                 x_t = x0

#             # FIX: clamp nới lỏng hơn (0.5 thay vì 0.3)
#             # 0.5 units ≈ 0.5×50×0.1°×111km = ~278km/6h (vẫn hợp lý cho bão mạnh)
#             x_t[:,:,:2] = x_t[:,:,:2].clamp(-0.5, 0.5)

#         return x_t   # [B, pred_len, 4]

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=5, ddim_steps=20):
#         """
#         Ensemble DDIM + Regression blend + Per-step steering correction.

#         Steering correction:
#         - Tính obs acceleration (thay đổi hướng) từ lịch sử
#         - Áp dụng acceleration đó vào predicted trajectory theo từng bước
#         - Tắt dần sau vài bước (bão thay đổi hướng chậm dần)
#         """
#         obs_t, obs_m = batch_list[0], batch_list[7]
#         lp, lm       = obs_t[-1], obs_m[-1]
#         T_pred       = self.pred_len
#         device       = lp.device

#         # ── Tính obs velocity và acceleration ────────────────────────────
#         if obs_t.shape[0] >= 2:
#             obs_vel = obs_t[-1] - obs_t[-2]   # [B, 2] last velocity
#         else:
#             obs_vel = torch.zeros_like(lp)

#         if obs_t.shape[0] >= 3:
#             obs_vel_prev = obs_t[-2] - obs_t[-3]
#             obs_accel    = obs_vel - obs_vel_prev   # [B, 2] acceleration (turning rate)
#         else:
#             obs_accel = torch.zeros_like(lp)

#         # ── DDIM ensemble ──────────────────────────────────────────────────
#         all_rels = []
#         trajs    = []
#         for _ in range(num_ensemble):
#             rel = self._ddim_once(batch_list, ddim_steps)
#             all_rels.append(rel)
#             t, _ = self.rel_to_abs(rel, lp, lm)
#             trajs.append(t)
#         ens_traj = torch.stack(trajs).mean(0)   # [T, B, 2]

#         # ── Per-step steering correction ──────────────────────────────────
#         # Tư tưởng: từng bước dự đoán, chúng ta kỳ vọng bão tiếp tục
#         # turning theo đúng acceleration quan sát được, nhưng giảm dần
#         #
#         # Predicted rel displacements (mean):
#         mean_rel = torch.stack(all_rels).mean(0)   # [B, T, 4]

#         # Tích lũy correction: mỗi bước thêm obs_accel × decay[step]
#         # decay: mạnh ở đầu, yếu ở cuối (bão có thể thay đổi hướng)
#         decay   = torch.exp(-torch.arange(T_pred, dtype=torch.float32,
#                                            device=device) * 0.15)   # [T]
#         # steering correction per step: [T, B, 2]
#         steer   = obs_accel.unsqueeze(0) * decay.view(T_pred, 1, 1) * 0.4

#         # Apply steering: adjust predicted displacements
#         corrected_rel = mean_rel.clone()
#         corrected_rel[:, :, :2] = mean_rel[:, :, :2] + steer.permute(1, 0, 2)

#         # Convert corrected rel → absolute
#         steer_traj, _ = self.rel_to_abs(corrected_rel, lp, lm)

#         # Blend ensemble với steering-corrected version
#         ens_final = 0.5 * ens_traj + 0.5 * steer_traj

#         # ── Direct Regression ─────────────────────────────────────────────
#         reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
#         reg_traj, reg_me = self.rel_to_abs(reg_rel, lp, lm)

#         # Apply same steering correction to regression output
#         reg_rel_c = reg_rel.clone()
#         reg_rel_c[:, :, :2] = reg_rel[:, :, :2] + steer.permute(1, 0, 2)
#         reg_traj_c, _ = self.rel_to_abs(reg_rel_c, lp, lm)

#         # ── Blend diffusion + regression ──────────────────────────────────
#         bw         = torch.sigmoid(self.denoiser.blend_logit).item()
#         final_traj = (1 - bw) * reg_traj_c + bw * ens_final

#         # ── pred_Me ───────────────────────────────────────────────────────
#         mes = []
#         for _ in range(max(1, num_ensemble//2)):
#             rel = self._ddim_once(batch_list, ddim_steps)
#             _, m = self.rel_to_abs(rel, lp, lm)
#             mes.append(m)
#         pred_me = torch.stack(mes).mean(0)

#         return final_traj, pred_me


# _______------------- version mới -----------------________
"""
TCNM/diffusion_model.py  ── v5
FIXES so với v4:
1. blend_logit có gradient trong get_loss() (trước chỉ dùng trong no_grad sample)
2. Clamp displacement nới lỏng hơn (0.5 thay vì 0.3) để không kẹt khi bão di chuyển nhanh
3. DDIM sampling ổn định hơn (không re-sample Me riêng, dùng chung rel)
4. Regression head weight tăng early training, giảm dần khi diffusion ổn định
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net


# ── Direct Regression Head ────────────────────────────────────────────────────
class DirectRegressionHead(nn.Module):
    """
    Dự báo displacement có khả năng học TURNING (thay đổi hướng).

    Key fix: input vào step_rnn bao gồm cả acceleration (delta_vel)
    chứ không chỉ velocity → model có thể học bão đang bắt đầu recurve.
    """
    def __init__(self, ctx_dim=128, pred_len=12):
        super().__init__()
        self.pred_len = pred_len

        # Encode obs: velocity + acceleration
        # input = [delta_pos(2), delta_Me(2), accel(2)] = 6 dims
        self.vel_encoder = nn.GRU(
            input_size=6, hidden_size=256, num_layers=2,
            batch_first=True, dropout=0.1
        )

        # step_rnn input: ctx + vel(2) + accel(2) + Me_vel(2) = ctx+6
        self.step_rnn  = nn.GRUCell(input_size=ctx_dim + 6, hidden_size=256)
        self.step_proj = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4)
        )
        # Predict acceleration correction (how much to turn)
        self.accel_proj = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2)
        )
        self.ctx_proj = nn.Linear(ctx_dim, ctx_dim)

    def forward(self, ctx, obs_traj, obs_Me):
        B      = ctx.shape[0]
        device = ctx.device

        # Build obs sequence: [vel, Me_vel, accel]
        if obs_traj.shape[0] >= 2:
            vel_seq    = obs_traj[1:] - obs_traj[:-1]       # [T-1, B, 2]
            Me_vel_seq = obs_Me[1:]   - obs_Me[:-1]
        else:
            vel_seq    = torch.zeros(1, B, 2, device=device)
            Me_vel_seq = torch.zeros(1, B, 2, device=device)

        if vel_seq.shape[0] >= 2:
            accel_seq = vel_seq[1:] - vel_seq[:-1]          # [T-2, B, 2]
            # pad to same length
            accel_seq = torch.cat([accel_seq[:1], accel_seq], dim=0)
        else:
            accel_seq = torch.zeros_like(vel_seq)

        # Take min length
        T_seq = min(vel_seq.shape[0], accel_seq.shape[0])
        obs_in = torch.cat([
            vel_seq[-T_seq:],
            Me_vel_seq[-T_seq:],
            accel_seq[-T_seq:]
        ], dim=-1).permute(1, 0, 2)   # [B, T_seq, 6]

        _, h = self.vel_encoder(obs_in)
        h    = h[-1]   # [B, 256]

        # Initial conditions: last velocity + last acceleration
        last_vel   = vel_seq[-1]      # [B, 2]
        last_Me_v  = Me_vel_seq[-1]   # [B, 2]
        if vel_seq.shape[0] >= 2:
            last_accel = vel_seq[-1] - vel_seq[-2]
        else:
            last_accel = torch.zeros(B, 2, device=device)

        ctx_p = self.ctx_proj(ctx)
        preds = []
        cur_vel   = last_vel
        cur_accel = last_accel
        cur_Me_v  = last_Me_v

        for _ in range(self.pred_len):
            inp = torch.cat([ctx_p, cur_vel, cur_accel, cur_Me_v], dim=-1)
            h   = self.step_rnn(inp, h)

            # Predict displacement and acceleration correction
            delta      = self.step_proj(h)         # [B, 4]
            accel_corr = self.accel_proj(h)         # [B, 2] — how to turn

            preds.append(delta)

            # Update velocity with learned acceleration
            cur_accel = accel_corr
            cur_vel   = delta[:, :2]
            cur_Me_v  = delta[:, 2:]

        return torch.stack(preds, dim=1)   # [B, pred_len, 4]


# ── Denoiser ──────────────────────────────────────────────────────────────────
class DeterministicDenoiser(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, ctx_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len
        self.ctx_dim  = ctx_dim

        self.spatial_enc = Unet3D(in_channel=1, out_channel=1)
        self.env_enc     = Env_net(obs_len=obs_len, d_model=64)
        self.obs_lstm    = nn.LSTM(input_size=4, hidden_size=128,
                                   num_layers=3, batch_first=True, dropout=0.2)

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        self.time_fc1   = nn.Linear(128, 256)
        self.time_fc2   = nn.Linear(256, 128)
        self.step_embed = nn.Linear(4, 128)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128, nhead=8, dim_feedforward=512,
            dropout=0.15, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

        # pos_enc: detect_pred_len() đọc key này
        self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

        self.reg_head   = DirectRegressionHead(ctx_dim=ctx_dim, pred_len=pred_len)

        # FIX: blend_logit phải tham gia vào loss để có gradient
        # Khởi đầu = 0 → sigmoid(0) = 0.5
        self.blend_logit = nn.Parameter(torch.zeros(1))

    def timestep_embedding(self, t, dim=128):
        device   = t.device
        half     = dim // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32, device=device)
                         * (-math.log(10000.0) / (half - 1)))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, 1)) if dim % 2 else emb

    def _extract_context(self, batch_list):
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        f_s = self.spatial_enc(image_obs).mean(dim=2)
        f_s = self.spatial_pool(f_s).flatten(1)          # [B,16]

        f_e, _, _ = self.env_enc(env_data, image_obs)    # [B,64]

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        _, (h_n, _) = self.obs_lstm(obs_in)
        f_h = h_n[-1]                                     # [B,128]

        ctx = torch.cat([f_s, f_e, f_h], dim=-1)
        ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
        ctx = self.ctx_drop(ctx)
        return self.ctx_fc2(ctx), obs_traj, obs_Me        # [B,128]

    def forward(self, x_t, t, batch_list):
        """Predict noise [B, pred_len, 4]"""
        device = x_t.device
        t = t.to(device)
        if t.dim() == 2:
            t = t.squeeze(1)

        ctx, obs_traj, obs_Me = self._extract_context(batch_list)

        t_emb = F.gelu(self.time_fc1(self.timestep_embedding(t)))
        t_emb = self.time_fc2(t_emb)

        x_emb  = self.step_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
        out    = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))

    def predict_direct(self, batch_list):
        ctx, obs_traj, obs_Me = self._extract_context(batch_list)
        return self.reg_head(ctx, obs_traj, obs_Me), ctx, obs_traj, obs_Me


# ── TCDiffusion ───────────────────────────────────────────────────────────────
class TCDiffusion(nn.Module):
    def __init__(self, pred_len=12, obs_len=8, num_steps=100, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.denoiser  = DeterministicDenoiser(pred_len, obs_len)

    # ── Noise schedule ────────────────────────────────────────────────────────
    def _cosine_beta_schedule(self, T, s=0.008, device='cpu'):
        x  = torch.linspace(0, T, T + 1, device=device)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        return torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.02)

    def _get_params(self, device):
        b  = self._cosine_beta_schedule(self.num_steps, device=device)
        a  = 1 - b
        ac = torch.cumprod(a, dim=0)
        return {'betas': b, 'alphas': a, 'ac': ac,
                'sqrt_ac': ac.sqrt(), 'sqrt_1mac': (1 - ac).sqrt()}

    # ── Relative helpers ──────────────────────────────────────────────────────
    @staticmethod
    def traj_to_rel(traj_gt, Me_gt, last_pos, last_Me):
        """Absolute → relative cumulative displacement [B, T, 4]"""
        tf = torch.cat([last_pos.unsqueeze(0), traj_gt], dim=0)
        mf = torch.cat([last_Me.unsqueeze(0),  Me_gt],   dim=0)
        return torch.cat([tf[1:]-tf[:-1], mf[1:]-mf[:-1]], dim=-1).permute(1,0,2)

    @staticmethod
    def rel_to_abs(rel, last_pos, last_Me):
        """Relative [B,T,4] → absolute [T,B,2], [T,B,2]"""
        d    = rel.permute(1, 0, 2)
        traj = last_pos.unsqueeze(0) + torch.cumsum(d[:,:,:2], dim=0)
        me   = last_Me.unsqueeze(0)  + torch.cumsum(d[:,:,2:], dim=0)
        return traj, me

    # ── Losses ────────────────────────────────────────────────────────────────
    def _dir_loss(self, pred, gt, last_pos):
        ref = last_pos.unsqueeze(0)
        return torch.clamp(0.7 - (F.normalize(gt-ref, p=2, dim=-1) *
                                   F.normalize(pred-ref, p=2, dim=-1)).sum(-1),
                           min=0).mean()

    def _smooth_loss(self, rel):
        if rel.shape[1] < 2: return rel.new_zeros(1).squeeze()
        return ((rel[:,1:,:2] - rel[:,:-1,:2])**2).mean()

    def _curvature_loss(self, pred_abs, gt_abs):
        """
        Penalize khi track dự đoán không uốn cong giống ground truth.
        Tính bằng cross-product của velocity vectors liên tiếp.
        pred_abs / gt_abs: [T, B, 2]
        """
        if pred_abs.shape[0] < 3:
            return pred_abs.new_zeros(1).squeeze()

        # Velocity vectors
        pred_v = pred_abs[1:] - pred_abs[:-1]   # [T-1, B, 2]
        gt_v   = gt_abs[1:]   - gt_abs[:-1]

        # Angular change (cross product z-component: vx*wy - vy*wx)
        pred_curl = pred_v[1:,:,0]*pred_v[:-1,:,1] - pred_v[1:,:,1]*pred_v[:-1,:,0]
        gt_curl   = gt_v[1:,:,0]  *gt_v[:-1,:,1]   - gt_v[1:,:,1]  *gt_v[:-1,:,0]

        # Penalize sign mismatch (uốn ngược chiều) và magnitude mismatch
        sign_loss = F.relu(-(pred_curl * gt_curl)).mean()           # wrong turn direction
        mag_loss  = F.mse_loss(pred_curl, gt_curl)                  # wrong curvature amount
        return sign_loss + 0.3 * mag_loss

    def _weighted_disp_loss(self, pred_abs, gt_abs):
        """
        L1 loss với weight tăng dần theo thời gian.
        Bước xa (48h, 72h) bị penalize nặng hơn → model học uốn cong đúng ở cuối.
        pred_abs / gt_abs: [T, B, 2]
        """
        T = pred_abs.shape[0]
        # Weight: từ 1.0 (bước đầu) → 2.5 (bước cuối)
        w = torch.linspace(1.0, 2.5, T, device=pred_abs.device).view(T, 1, 1)
        return (w * (pred_abs - gt_abs).abs()).mean()

    # ── Training loss ─────────────────────────────────────────────────────────
    def get_loss(self, batch_list):
        traj_gt = batch_list[1]   # [T, B, 2]
        Me_gt   = batch_list[8]
        obs     = batch_list[0]
        obs_Me  = batch_list[7]

        B      = traj_gt.shape[1]
        device = traj_gt.device
        lp, lm = obs[-1], obs_Me[-1]

        x_0 = self.traj_to_rel(traj_gt, Me_gt, lp, lm)   # [B,T,4]

        p   = self._get_params(device)
        t   = torch.randint(0, self.num_steps, (B,), device=device).long()
        eps = torch.randn_like(x_0) * 0.1
        sac   = p['sqrt_ac'][t].view(B,1,1)
        s1mac = p['sqrt_1mac'][t].view(B,1,1)
        x_t   = sac * x_0 + s1mac * eps

        pred_eps = self.denoiser(x_t, t, batch_list)
        mse      = F.mse_loss(pred_eps, eps)

        pred_x0 = (x_t - s1mac * pred_eps) / (sac + 1e-8)
        pred_abs, _ = self.rel_to_abs(pred_x0, lp, lm)

        dir_l  = self._dir_loss(pred_abs, traj_gt, lp)
        smt_l  = self._smooth_loss(pred_x0)
        disp_l = self._weighted_disp_loss(pred_abs, traj_gt)  # weighted L1
        curv_l = self._curvature_loss(pred_abs, traj_gt)       # curvature

        diff_l = mse + 2.0*dir_l + 0.5*smt_l + 1.0*disp_l + 1.5*curv_l

        # ── Regression head ───────────────────────────────────────────────
        reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
        reg_abs, _       = self.rel_to_abs(reg_rel, lp, lm)
        reg_l = (self._weighted_disp_loss(reg_abs, traj_gt)
                 + self._dir_loss(reg_abs, traj_gt, lp)
                 + self._smooth_loss(reg_rel)
                 + self._curvature_loss(reg_abs, traj_gt))

        # ── Blend loss có gradient ────────────────────────────────────────
        blend_w = torch.sigmoid(self.denoiser.blend_logit)
        blended_abs = (1 - blend_w) * reg_abs + blend_w * pred_abs
        blend_l = self._weighted_disp_loss(blended_abs, traj_gt)

        blend_reg = ((blend_w - 0.5)**2) * 0.01

        return diff_l + 0.5*reg_l + 0.3*blend_l + blend_reg

    # ── DDIM Sampling ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ddim_once(self, batch_list, ddim_steps=20):
        B      = batch_list[0].shape[1]
        device = batch_list[0].device
        p      = self._get_params(device)

        obs = batch_list[0]
        # FIX: scale noise theo observed velocity thực tế
        # vel ≈ displacement/step trong normalized units
        # Bão WIPHA di chuyển ~50km/6h = 50/(50*11.1) ≈ 0.09 normalized units/step
        # scale = 1.0 để noise đủ lớn cho sampling khám phá được trajectories xa
        if obs.shape[0] >= 2:
            vel = (obs[-1] - obs[-2]).abs().mean().item()
        else:
            vel = 0.05
        scale = max(vel * 3.0, 0.05)   # FIX: tăng từ 0.5 → 3.0

        x_t = torch.randn(B, self.pred_len, 4, device=device) * scale

        steps = torch.linspace(self.num_steps-1, 0, ddim_steps,
                               dtype=torch.long, device=device)

        for i, step in enumerate(steps):
            sv      = step.item()
            t_b     = torch.full((B,), sv, device=device, dtype=torch.long)
            eps     = self.denoiser(x_t, t_b, batch_list)
            sac     = p['sqrt_ac'][sv]
            s1mac   = p['sqrt_1mac'][sv]
            # FIX: nới lỏng clamp từ 0.5 → 1.5
            # 1.5 units/step × 12 steps × 50 × 11.1km ≈ 10,000km tổng
            # (bão thực tế ~700km/72h = ~58km/6h ≈ 0.10 units/step → tổng ~1.2 units)
            x0      = ((x_t - s1mac * eps) / (sac + 1e-8)).clamp(-1.5, 1.5)

            if i < len(steps) - 1:
                ns   = steps[i+1].item()
                x_t  = p['sqrt_ac'][ns] * x0 + p['sqrt_1mac'][ns] * eps
            else:
                x_t = x0

            x_t[:,:,:2] = x_t[:,:,:2].clamp(-1.5, 1.5)  # FIX: nới lỏng

        return x_t   # [B, pred_len, 4]

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=5, ddim_steps=20):
        """
        Ensemble DDIM + Regression blend + Per-step steering correction.

        Steering correction:
        - Tính obs acceleration (thay đổi hướng) từ lịch sử
        - Áp dụng acceleration đó vào predicted trajectory theo từng bước
        - Tắt dần sau vài bước (bão thay đổi hướng chậm dần)
        """
        obs_t, obs_m = batch_list[0], batch_list[7]
        lp, lm       = obs_t[-1], obs_m[-1]
        T_pred       = self.pred_len
        device       = lp.device

        # ── Tính obs velocity và acceleration ────────────────────────────
        if obs_t.shape[0] >= 2:
            obs_vel = obs_t[-1] - obs_t[-2]   # [B, 2] last velocity
        else:
            obs_vel = torch.zeros_like(lp)

        if obs_t.shape[0] >= 3:
            obs_vel_prev = obs_t[-2] - obs_t[-3]
            obs_accel    = obs_vel - obs_vel_prev   # [B, 2] acceleration (turning rate)
        else:
            obs_accel = torch.zeros_like(lp)

        # ── DDIM ensemble ──────────────────────────────────────────────────
        all_rels = []
        trajs    = []
        for _ in range(num_ensemble):
            rel = self._ddim_once(batch_list, ddim_steps)
            all_rels.append(rel)
            t, _ = self.rel_to_abs(rel, lp, lm)
            trajs.append(t)
        ens_traj = torch.stack(trajs).mean(0)   # [T, B, 2]

        # ── Velocity anchoring ────────────────────────────────────────────
        # Vấn đề "co cụm": model predict displacement quá nhỏ do regression to mean.
        # Fix: neo predicted trajectory vào observed velocity.
        # Nếu bão đang đi 0.09 units/step thì predicted track cũng phải dãn ra tương đương.
        mean_rel = torch.stack(all_rels).mean(0)   # [B, T, 4]

        # Tính mean predicted velocity (bước đầu)
        pred_vel_0 = mean_rel[:, 0, :2]           # [B, 2]
        pred_speed = pred_vel_0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        obs_speed  = obs_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Nếu predicted speed < 50% obs speed → scale up
        speed_ratio = (obs_speed / pred_speed).clamp(0.5, 3.0)  # không scale quá mạnh

        # Scale displacement của ensemble
        ens_rel_scaled = mean_rel.clone()
        ens_rel_scaled[:, :, :2] = mean_rel[:, :, :2] * speed_ratio.unsqueeze(1)
        ens_traj_scaled, _ = self.rel_to_abs(ens_rel_scaled, lp, lm)

        # Blend ensemble gốc và scaled version
        ens_final = 0.4 * ens_traj + 0.6 * ens_traj_scaled

        # ── Per-step steering correction (giữ nguyên) ────────────────────
        decay = torch.exp(-torch.arange(T_pred, dtype=torch.float32,
                                         device=device) * 0.15)
        steer = obs_accel.unsqueeze(0) * decay.view(T_pred, 1, 1) * 0.4
        corrected_rel = ens_rel_scaled.clone()
        corrected_rel[:, :, :2] = ens_rel_scaled[:, :, :2] + steer.permute(1, 0, 2)
        steer_traj, _ = self.rel_to_abs(corrected_rel, lp, lm)
        ens_final = 0.5 * ens_final + 0.5 * steer_traj

        # ── Direct Regression ─────────────────────────────────────────────
        reg_rel, _, _, _ = self.denoiser.predict_direct(batch_list)
        reg_traj, reg_me = self.rel_to_abs(reg_rel, lp, lm)

        # Scale regression output tương tự
        reg_vel_0   = reg_rel[:, 0, :2]
        reg_speed   = reg_vel_0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        reg_ratio   = (obs_speed / reg_speed).clamp(0.5, 3.0)
        reg_rel_c   = reg_rel.clone()
        reg_rel_c[:, :, :2] = reg_rel[:, :, :2] * reg_ratio.unsqueeze(1)
        reg_traj_c, _ = self.rel_to_abs(reg_rel_c, lp, lm)

        # ── Blend diffusion + regression ──────────────────────────────────
        bw         = torch.sigmoid(self.denoiser.blend_logit).item()
        final_traj = (1 - bw) * reg_traj_c + bw * ens_final

        # ── pred_Me ───────────────────────────────────────────────────────
        mes = []
        for _ in range(max(1, num_ensemble//2)):
            rel = self._ddim_once(batch_list, ddim_steps)
            _, m = self.rel_to_abs(rel, lp, lm)
            mes.append(m)
        pred_me = torch.stack(mes).mean(0)

        return final_traj, pred_me