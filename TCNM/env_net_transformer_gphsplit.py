"""
TCNM/env_net_transformer_gphsplit.py
Handles None env_data and missing keys robustly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Env_net(nn.Module):
    """Environmental Transformer with robust None handling"""
    def __init__(self, obs_len=8, embed_dim=16, d_model=64):
        super().__init__()
        self.obs_len = obs_len
        self.d_model = d_model
        self.embed_dim = embed_dim

        self.data_embed = nn.ModuleDict({
            'wind':                 nn.Linear(1, embed_dim),
            'intensity_class':      nn.Linear(6, embed_dim),
            'move_velocity':        nn.Linear(1, embed_dim),
            'month':               nn.Linear(12, embed_dim),
            'location_long':       nn.Linear(36, embed_dim),
            'location_lat':        nn.Linear(12, embed_dim),
            'history_direction12': nn.Linear(8, embed_dim),
            'history_direction24': nn.Linear(8, embed_dim),
            'history_inte_change24': nn.Linear(4, embed_dim),
        })

        self.GPH_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 2))
        )  # output: 128*2*2 = 512

        env_f_total = len(self.data_embed) * embed_dim + 512  # 9*16 + 512 = 656
        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_total, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_encoder = nn.Parameter(torch.randn(1, obs_len, d_model) * 0.02)

    def forward(self, env_data, gph):
        """
        Args:
            env_data: dict or None
            gph: [B, C, T, H, W] or [B, T, H, W] or [B, 1, T, H, W]
        Returns:
            (context_vec [B, d_model], 0, 0)
        """
        device = gph.device

        # --- Normalize GPH shape to [B, 1, T, H, W] ---
        if gph.dim() == 4:          # [B, T, H, W]
            gph = gph.unsqueeze(1)
        elif gph.dim() == 5 and gph.shape[1] != 1:
            # [B, T, 1, H, W] → [B, 1, T, H, W]
            gph = gph.permute(0, 2, 1, 3, 4)

        B, C, T, H, W = gph.shape
        seq_len = T

        # GPH features: [B*T, 1, H, W] → [B, T, 512]
        gph_flat = gph.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        gph_f = self.GPH_embed(gph_flat).view(B, T, 512)

        # --- Env features ---
        embed_list = []
        for key, layer in self.data_embed.items():
            in_f = layer.in_features
            if env_data is None or not isinstance(env_data, dict) or key not in env_data or env_data[key] is None:
                embed_list.append(torch.zeros(B, T, self.embed_dim, device=device))
                continue

            val = env_data[key]
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float, device=device)
            else:
                val = val.float().to(device)

            # Normalize to [B, T, in_f]
            if val.dim() == 1:
                val = val.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            elif val.dim() == 2:
                # [B, in_f] or [B, T]
                if val.shape == (B, in_f):
                    val = val.unsqueeze(1).expand(-1, T, -1)
                elif val.shape[0] == B:
                    val = val.unsqueeze(-1) if val.dim() == 2 else val
                    val = val.unsqueeze(1).expand(-1, T, -1)
                else:
                    val = torch.zeros(B, T, in_f, device=device)
            elif val.dim() == 3:
                # [B, T, in_f] or [B, 1, in_f]
                if val.shape[1] != T:
                    val = val.expand(-1, T, -1) if val.shape[1] == 1 else F.pad(val, (0,0,0,T-val.shape[1]))[:, :T]

            # Adjust feature dim
            if val.shape[-1] != in_f:
                if val.shape[-1] < in_f:
                    val = F.pad(val, (0, in_f - val.shape[-1]))
                else:
                    val = val[..., :in_f]

            try:
                embed_list.append(layer(val))
            except Exception as e:
                embed_list.append(torch.zeros(B, T, self.embed_dim, device=device))

        embed_list.append(gph_f)
        combined = torch.cat(embed_list, dim=2)   # [B, T, 9*16 + 512]
        feature_in = self.evn_extract(combined)   # [B, T, d_model]

        # Positional encoding
        if self.pos_encoder.shape[1] == T:
            feature_in = feature_in + self.pos_encoder
        else:
            pe = F.interpolate(self.pos_encoder.permute(0,2,1), size=T, mode='linear', align_corners=False).permute(0,2,1)
            feature_in = feature_in + pe

        output = self.encoder(feature_in)
        return output[:, -1, :], 0, 0