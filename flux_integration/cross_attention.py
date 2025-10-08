"""
Cross-Attention Module for Decomposition Guidance

Injects decomposition features (intensity, material, context) into FLUX transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecompositionCrossAttention(nn.Module):
    """
    Cross-attention layer that injects decomposition features into FLUX.

    Query: FLUX image features
    Key/Value: Decomposition features (intensity + material + context)
    """

    def __init__(
        self,
        image_dim=3072,  # FLUX hidden dimension
        decomp_dim=128,  # Decomposition feature dimension after encoding
        num_heads=8,
        dropout=0.0
    ):
        """
        Args:
            image_dim: Dimension of FLUX image features
            decomp_dim: Dimension of encoded decomposition features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.image_dim = image_dim
        self.decomp_dim = decomp_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads

        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"

        # Layer normalization
        self.norm_image = nn.LayerNorm(image_dim)
        self.norm_decomp = nn.LayerNorm(decomp_dim)

        # Projection layers
        self.q_proj = nn.Linear(image_dim, image_dim)
        self.k_proj = nn.Linear(decomp_dim, image_dim)
        self.v_proj = nn.Linear(decomp_dim, image_dim)
        self.out_proj = nn.Linear(image_dim, image_dim)

        self.dropout = nn.Dropout(dropout)

        # Learnable scale parameter (start small to avoid disrupting pretrained FLUX)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, image_feat, decomp_feat, attention_mask=None):
        """
        Args:
            image_feat: [B, N_img, image_dim] FLUX image features
            decomp_feat: [B, N_decomp, decomp_dim] Encoded decomposition features
            attention_mask: Optional [B, N_img, N_decomp] attention mask

        Returns:
            output: [B, N_img, image_dim] Enhanced features
        """
        B, N_img, _ = image_feat.shape
        N_decomp = decomp_feat.shape[1]

        # Normalize inputs
        image_norm = self.norm_image(image_feat)
        decomp_norm = self.norm_decomp(decomp_feat)

        # Project to Q, K, V
        Q = self.q_proj(image_norm)  # [B, N_img, image_dim]
        K = self.k_proj(decomp_norm)  # [B, N_decomp, image_dim]
        V = self.v_proj(decomp_norm)  # [B, N_decomp, image_dim]

        # Reshape for multi-head attention
        Q = Q.view(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_img, head_dim]
        K = K.view(B, N_decomp, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_decomp, head_dim]
        V = V.view(B, N_decomp, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_decomp, head_dim]

        # Scaled dot-product attention
        scale_factor = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale_factor  # [B, num_heads, N_img, N_decomp]

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1)  # Broadcast over heads

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, N_img, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_img, self.image_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection with learnable scale
        output = image_feat + self.scale * attn_output

        return output


class DecompositionEncoder(nn.Module):
    """
    Encodes decomposition features (intensity, material, context) into a unified representation
    suitable for cross-attention with FLUX.
    """

    def __init__(
        self,
        num_material_classes=32,
        context_channels=8,
        hidden_dim=128,
        output_dim=128
    ):
        """
        Args:
            num_material_classes: Number of material classes
            context_channels: Number of context channels
            hidden_dim: Hidden dimension for processing
            output_dim: Output dimension for cross-attention
        """
        super().__init__()

        # Process intensity (1 channel)
        self.intensity_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )

        # Process material (num_classes channels after softmax)
        self.material_encoder = nn.Sequential(
            nn.Conv2d(num_material_classes, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Process context (context_channels)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(context_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, intensity, material_logits, context):
        """
        Args:
            intensity: [B, 1, H, W]
            material_logits: [B, num_classes, H, W]
            context: [B, context_channels, H, W]

        Returns:
            encoded: [B, N, output_dim] where N = H * W
        """
        B, _, H, W = intensity.shape

        # Encode each component
        intensity_feat = self.intensity_encoder(intensity)  # [B, hidden_dim/4, H, W]

        material_probs = F.softmax(material_logits, dim=1)
        material_feat = self.material_encoder(material_probs)  # [B, hidden_dim/2, H, W]

        context_feat = self.context_encoder(context)  # [B, hidden_dim/4, H, W]

        # Concatenate along channel dimension
        combined = torch.cat([intensity_feat, material_feat, context_feat], dim=1)  # [B, hidden_dim, H, W]

        # Fusion
        encoded = self.fusion(combined)  # [B, output_dim, H, W]

        # Reshape to sequence: [B, N, output_dim]
        encoded = encoded.flatten(2).transpose(1, 2)  # [B, H*W, output_dim]

        return encoded


if __name__ == '__main__':
    # Test the modules
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test DecompositionEncoder
    encoder = DecompositionEncoder(
        num_material_classes=32,
        context_channels=8,
        hidden_dim=128,
        output_dim=128
    ).to(device)

    B, H, W = 2, 32, 32
    intensity = torch.rand(B, 1, H, W).to(device)
    material_logits = torch.randn(B, 32, H, W).to(device)
    context = torch.randn(B, 8, H, W).to(device)

    encoded = encoder(intensity, material_logits, context)
    print(f"Encoded shape: {encoded.shape}")  # Should be [2, 1024, 128]

    # Test DecompositionCrossAttention
    cross_attn = DecompositionCrossAttention(
        image_dim=3072,
        decomp_dim=128,
        num_heads=8
    ).to(device)

    N_img = 256
    image_feat = torch.randn(B, N_img, 3072).to(device)

    output = cross_attn(image_feat, encoded)
    print(f"Output shape: {output.shape}")  # Should be [2, 256, 3072]

    print("\nCross-attention module test successful!")
