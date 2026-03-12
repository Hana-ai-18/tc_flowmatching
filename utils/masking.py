# """TCNM/utils/masking.py - Attention masks"""
# import torch

# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)
#     @property
#     def mask(self):
#         return self._mask

# class ProbMask():
#     def __init__(self, B, H, L, index, scores, device="cpu"):
#         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
#         batch_idx = torch.arange(B, device=device)[:, None, None]
#         head_idx = torch.arange(H, device=device)[None, :, None]
#         indicator = _mask_ex[batch_idx, head_idx, index, :]
#         self._mask = indicator.view(scores.shape)
#     @property
#     def mask(self):
#         return self._mask

"""TCNM/utils/masking.py - Attention masks for Transformer components"""
import torch


class TriangularCausalMask:
    """
    Upper-triangular causal mask — prevents attending to future positions.
    Shape: [B, 1, L, L]  (True = masked / ignored)
    """
    def __init__(self, B: int, L: int, device: str = "cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device),
                diagonal=1
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    """
    ProbSparse attention mask used in Informer-style Transformers.
    Selects top-k queries and masks the rest.

    Args:
        B      : batch size
        H      : number of heads
        L      : query length
        index  : selected query indices [B, H, top_k]
        scores : attention score tensor  [B, H, top_k, L_kv]
        device : target device
    """
    def __init__(
        self,
        B:      int,
        H:      int,
        L:      int,
        index:  torch.Tensor,
        scores: torch.Tensor,
        device: str = "cpu",
    ):
        L_kv = scores.shape[-1]

        # Full causal mask [L, L_kv]
        _mask    = torch.ones(L, L_kv, dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, L_kv)  # [B, H, L, L_kv]

        # Gather rows corresponding to selected queries
        batch_idx = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
        head_idx  = torch.arange(H, device=device)[None, :, None]  # [1, H, 1]
        indicator = _mask_ex[batch_idx, head_idx, index, :]         # [B, H, top_k, L_kv]

        self._mask = indicator.view(scores.shape)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask