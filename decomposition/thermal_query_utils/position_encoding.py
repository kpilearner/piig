"""
Position Encoding Module

Adapted from DDColor's position encoding implementation.
Provides sinusoidal position embeddings for spatial features.
"""

import math
import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    Sine-based positional encoding for 2D spatial features.

    This is standard positional encoding used in Transformer models,
    adapted for 2D feature maps.

    Args:
        num_pos_feats: Number of positional features (half of final dim)
        temperature: Temperature for sine frequency
        normalize: Whether to normalize coordinates to [0, 1]
        scale: Scale factor if normalizing
    """

    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=True,
        scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
            mask: Optional mask of shape [B, H, W]

        Returns:
            pos: Positional encoding of shape [B, num_pos_feats*2, H, W]
        """
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool
            )

        # Invert mask (0 = valid position, 1 = padding)
        not_mask = ~mask

        # Cumulative sum to get y, x coordinates
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Compute sine/cosine for x and y coordinates
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Stack sine and cosine: [B, H, W, num_pos_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        # Concatenate x and y embeddings: [B, H, W, num_pos_feats*2]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
