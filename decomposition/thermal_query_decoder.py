"""
Thermal Query Decoder

Inspired by DDColor's learnable color queries approach,
adapted for infrared thermal image generation from RGB.

Core Idea:
- Learn N "thermal patterns" as embeddings (similar to DETR's object queries)
- Use multi-scale cross-attention to extract thermal information from RGB features
- Use self-attention for queries to achieve global thermal consistency
- No explicit material classification needed - queries implicitly learn thermal modes

Reference:
- DDColor: https://github.com/piddnad/DDColor
- DETR: https://github.com/facebookresearch/detr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .thermal_query_utils import (
    PositionEmbeddingSine,
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP
)


class ThermalQueryDecoder(nn.Module):
    """
    Thermal Query Decoder with learnable thermal pattern queries.

    Architecture:
        RGB Image → ResNet Encoder → Multi-scale features
                                            ↓
                        Thermal Queries (learnable embeddings)
                                            ↓
                        Multi-scale Cross-Attention
                        + Self-Attention (9 layers)
                                            ↓
                        Thermal Embeddings → Infrared Output

    Args:
        backbone: ResNet backbone ('resnet18', 'resnet50', etc.)
        pretrained: Use ImageNet pretrained weights
        num_thermal_queries: Number of learnable thermal pattern queries
        hidden_dim: Hidden dimension for queries and attention
        num_scales: Number of multi-scale features to use
        dec_layers: Number of decoder layers
        nheads: Number of attention heads
        dim_feedforward: FFN hidden dimension
    """

    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        num_thermal_queries=256,
        hidden_dim=256,
        num_scales=3,
        dec_layers=9,
        nheads=8,
        dim_feedforward=2048,
        pre_norm=False,
    ):
        super().__init__()

        self.num_thermal_queries = num_thermal_queries
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.num_layers = dec_layers

        # ========== Shared ResNet Encoder ==========
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feat_dims = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feat_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract ResNet layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4 resolution
        self.layer2 = resnet.layer2  # 1/8 resolution
        self.layer3 = resnet.layer3  # 1/16 resolution
        self.layer4 = resnet.layer4  # 1/32 resolution

        # Select which layers to use for multi-scale features
        # Typically use last 3 layers: layer2, layer3, layer4
        self.in_channels = feat_dims[-num_scales:]  # e.g., [512, 1024, 2048] for ResNet50

        # ========== Positional Encoding ==========
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # ========== Level Embedding (for multi-scale) ==========
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        # ========== Input Projections (multi-scale features → hidden_dim) ==========
        self.input_proj = nn.ModuleList()
        for i in range(num_scales):
            self.input_proj.append(
                nn.Conv2d(self.in_channels[i], hidden_dim, kernel_size=1)
            )
            # Initialize projection layers
            nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
            if self.input_proj[-1].bias is not None:
                nn.init.constant_(self.input_proj[-1].bias, 0)

        # ========== Learnable Thermal Queries ==========
        # Query features: what the queries "look for"
        self.query_feat = nn.Embedding(num_thermal_queries, hidden_dim)
        # Query positional encoding: where the queries "focus"
        self.query_embed = nn.Embedding(num_thermal_queries, hidden_dim)

        # ========== Transformer Decoder Layers ==========
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # ========== Output Projection ==========
        # Project thermal queries to final thermal representation
        # Output: [B, num_queries, thermal_embed_dim]
        # Then use einsum to map to spatial dimension
        self.thermal_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Final conv to produce single-channel infrared output
        self.final_conv = nn.Conv2d(num_thermal_queries, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0)

    def forward_encoder(self, rgb):
        """
        Extract multi-scale features from RGB image.

        Args:
            rgb: [B, 3, H, W] RGB image (ImageNet normalized)

        Returns:
            multi_scale_feats: List of [x2, x3, x4] feature maps
        """
        # Initial conv
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x1 = self.layer1(x)   # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # Return last num_scales features
        all_feats = [x1, x2, x3, x4]
        multi_scale_feats = all_feats[-self.num_scales:]

        return multi_scale_feats

    def forward(self, rgb):
        """
        Forward pass: RGB → Infrared

        Args:
            rgb: [B, 3, H, W] RGB image

        Returns:
            infrared: [B, 1, H, W] Infrared output
        """
        B, _, H, W = rgb.shape

        # ========== 1. Extract Multi-scale Features ==========
        multi_scale_feats = self.forward_encoder(rgb)

        # ========== 2. Prepare Multi-scale Features for Transformer ==========
        src = []  # Feature sequences
        pos = []  # Positional encodings

        for i, feat in enumerate(multi_scale_feats):
            # Positional encoding for this scale
            pos_encoding = self.pe_layer(feat, None).flatten(2)  # [B, C, H*W]

            # Project feature + add level embedding
            projected_feat = self.input_proj[i](feat).flatten(2)  # [B, hidden_dim, H*W]
            projected_feat = projected_feat + self.level_embed.weight[i][None, :, None]

            # Transpose to [H*W, B, hidden_dim] for Transformer
            pos.append(pos_encoding.permute(2, 0, 1))
            src.append(projected_feat.permute(2, 0, 1))

        # ========== 3. Initialize Thermal Queries ==========
        # Query embeddings: [num_queries, B, hidden_dim]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)

        # ========== 4. Transformer Decoder (9 layers) ==========
        for i in range(self.num_layers):
            level_index = i % self.num_scales  # Cycle through scales

            # Cross-attention: queries ← image features
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed
            )

            # Self-attention: queries ↔ queries (global consistency)
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

        # ========== 5. Output Processing ==========
        # Normalize decoder output
        decoder_output = self.decoder_norm(output)  # [num_queries, B, hidden_dim]
        decoder_output = decoder_output.transpose(0, 1)  # [B, num_queries, hidden_dim]

        # Project to thermal embeddings
        thermal_embed = self.thermal_embed(decoder_output)  # [B, num_queries, hidden_dim]

        # Use einsum to map queries to spatial dimensions
        # We'll use the last feature map as the spatial reference
        last_feat = multi_scale_feats[-1]  # [B, C, H', W']
        H_feat, W_feat = last_feat.shape[2], last_feat.shape[3]

        # Project last_feat to hidden_dim for compatibility
        spatial_feat = self.input_proj[-1](last_feat)  # [B, hidden_dim, H', W']

        # Compute attention between queries and spatial features
        # thermal_embed: [B, num_queries, hidden_dim]
        # spatial_feat: [B, hidden_dim, H', W']
        out = torch.einsum("bqc,bchw->bqhw", thermal_embed, spatial_feat)
        # out: [B, num_queries, H', W']

        # Apply final conv to aggregate queries → single channel
        out = self.final_conv(out)  # [B, 1, H', W']

        # Upsample to original resolution
        infrared = F.interpolate(
            out,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Apply sigmoid to ensure [0, 1] range
        infrared = torch.sigmoid(infrared)

        return infrared


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Testing ThermalQueryDecoder")
    print("=" * 70)

    # Create model
    model = ThermalQueryDecoder(
        backbone='resnet18',  # Use ResNet18 for faster testing
        pretrained=False,
        num_thermal_queries=256,
        hidden_dim=256,
        num_scales=3,
        dec_layers=9,
        nheads=8,
    ).to(device)

    # Test input
    rgb = torch.randn(2, 3, 256, 256).to(device)

    print(f"\nInput RGB shape: {rgb.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Forward pass
    with torch.no_grad():
        infrared = model(rgb)

    print(f"\nOutput Infrared shape: {infrared.shape}")
    print(f"Output range: [{infrared.min():.3f}, {infrared.max():.3f}]")

    print("\n" + "=" * 70)
    print("✅ ThermalQueryDecoder test passed!")
    print("=" * 70)
