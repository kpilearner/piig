"""
Multi-Task Decomposition Network

This module implements a physics-inspired decomposition of RGB images into
three task-relevant representations:
- Intensity: Resembling temperature distribution (high values = hot regions)
- Material: Resembling material types (32 discrete classes)
- Context: Resembling environmental context (8-channel features)

NOTE: This is NOT physical decomposition. We learn representations that
RESEMBLE physics quantities as an inductive bias for better infrared generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskDecompositionNet(nn.Module):
    """
    Multi-task decomposition network with shared ResNet encoder
    and three task-specific decoder heads.
    """

    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        num_material_classes=32,
        context_channels=8
    ):
        """
        Args:
            backbone: ResNet backbone architecture ('resnet18', 'resnet50', etc.)
            pretrained: Whether to use ImageNet pretrained weights
            num_material_classes: Number of material classes for material head
            context_channels: Number of context feature channels
        """
        super().__init__()

        self.num_material_classes = num_material_classes
        self.context_channels = context_channels

        # Shared ResNet encoder (remove final FC layer)
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        # Three task-specific decoder heads
        self.intensity_head = IntensityDecoder(feat_dim)
        self.material_head = MaterialDecoder(feat_dim, num_material_classes)
        self.context_head = ContextDecoder(feat_dim, context_channels)

    def forward(self, rgb):
        """
        Args:
            rgb: [B, 3, H, W] RGB image

        Returns:
            intensity: [B, 1, H, W] Intensity map (resembling temperature)
            material: [B, num_classes, H, W] Material logits
            context: [B, context_channels, H, W] Context features
        """
        # Shared encoder
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)   # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # Three decoder heads
        intensity = self.intensity_head(x4, x3, x2, x1)
        material = self.material_head(x4, x3, x2, x1)
        context = self.context_head(x4, x3, x2, x1)

        return intensity, material, context


class IntensityDecoder(nn.Module):
    """Decoder for intensity prediction (resembling temperature distribution)"""

    def __init__(self, feat_dim):
        super().__init__()

        # Progressive upsampling with skip connections
        if feat_dim == 2048:  # ResNet50
            self.up1 = self._make_decoder_block(2048, 1024)
            self.up2 = self._make_decoder_block(1024, 512)
            self.up3 = self._make_decoder_block(512, 256)
            self.up4 = self._make_decoder_block(256, 64)
        else:  # ResNet18
            self.up1 = self._make_decoder_block(512, 256)
            self.up2 = self._make_decoder_block(256, 128)
            self.up3 = self._make_decoder_block(128, 64)
            self.up4 = self._make_decoder_block(64, 64)

        # Final 1x1 conv to single channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x4, x3, x2, x1):
        x = self.up1(x4)  # 1/16
        x = self.up2(x)   # 1/8
        x = self.up3(x)   # 1/4
        x = self.up4(x)   # 1/2

        # Upsample to original resolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        intensity = torch.sigmoid(self.final_conv(x))  # [0, 1] range

        return intensity


class MaterialDecoder(nn.Module):
    """Decoder for material classification (resembling material types)"""

    def __init__(self, feat_dim, num_classes):
        super().__init__()

        # Progressive upsampling
        if feat_dim == 2048:  # ResNet50
            self.up1 = self._make_decoder_block(2048, 1024)
            self.up2 = self._make_decoder_block(1024, 512)
            self.up3 = self._make_decoder_block(512, 256)
            self.up4 = self._make_decoder_block(256, 128)
        else:  # ResNet18
            self.up1 = self._make_decoder_block(512, 256)
            self.up2 = self._make_decoder_block(256, 128)
            self.up3 = self._make_decoder_block(128, 64)
            self.up4 = self._make_decoder_block(64, 128)

        # Final classification layer
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x4, x3, x2, x1):
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        material_logits = self.final_conv(x)  # [B, num_classes, H, W]

        return material_logits


class ContextDecoder(nn.Module):
    """Decoder for context features (environmental factors)"""

    def __init__(self, feat_dim, context_channels):
        super().__init__()

        # Progressive upsampling
        if feat_dim == 2048:  # ResNet50
            self.up1 = self._make_decoder_block(2048, 1024)
            self.up2 = self._make_decoder_block(1024, 512)
            self.up3 = self._make_decoder_block(512, 256)
            self.up4 = self._make_decoder_block(256, 64)
        else:  # ResNet18
            self.up1 = self._make_decoder_block(512, 256)
            self.up2 = self._make_decoder_block(256, 128)
            self.up3 = self._make_decoder_block(128, 64)
            self.up4 = self._make_decoder_block(64, 64)

        # Final conv for context features
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, context_channels, kernel_size=1),
            nn.Tanh()  # [-1, 1] range for flexibility
        )

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x4, x3, x2, x1):
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        context = self.final_conv(x)  # [B, context_channels, H, W]

        return context


class PhysicsInspiredFusion(nn.Module):
    """
    Physics-inspired fusion module that combines three decomposed components.

    This is NOT a strict physics formula (like Stefan-Boltzmann law).
    Instead, it's a learnable fusion that RESEMBLES physics:
    - Intensity and material interact (like temperature and emissivity)
    - Context modulates the result (like environmental factors)

    The fusion is learned from data, not hard-coded physics.
    """

    def __init__(self, num_material_classes=32, context_channels=8, hidden_dim=64):
        super().__init__()

        # Learnable material embeddings (resembling emissivity profiles)
        self.material_embeddings = nn.Parameter(
            torch.randn(num_material_classes, hidden_dim)
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv2d(1 + hidden_dim + context_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, intensity, material_logits, context):
        """
        Args:
            intensity: [B, 1, H, W]
            material_logits: [B, num_classes, H, W]
            context: [B, context_channels, H, W]

        Returns:
            fused_output: [B, 1, H, W] Fused representation
        """
        B, _, H, W = intensity.shape

        # Soft material assignment
        material_probs = F.softmax(material_logits, dim=1)  # [B, C, H, W]

        # Get material features via weighted sum of embeddings
        # Reshape for matrix multiplication
        material_probs_flat = material_probs.permute(0, 2, 3, 1).reshape(B*H*W, -1)  # [BHW, C]
        material_feat_flat = torch.matmul(
            material_probs_flat,
            self.material_embeddings
        )  # [BHW, hidden_dim]
        material_feat = material_feat_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]

        # Concatenate all components
        fused_input = torch.cat([intensity, material_feat, context], dim=1)

        # Learnable fusion (physics-inspired but not strict physics)
        fused_output = self.fusion_net(fused_input)

        return fused_output


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskDecompositionNet(backbone='resnet18').to(device)
    fusion = PhysicsInspiredFusion().to(device)

    # Test input
    rgb = torch.randn(2, 3, 256, 256).to(device)

    # Forward pass
    intensity, material, context = model(rgb)
    print(f"Intensity shape: {intensity.shape}")
    print(f"Material shape: {material.shape}")
    print(f"Context shape: {context.shape}")

    # Fusion
    fused = fusion(intensity, material, context)
    print(f"Fused shape: {fused.shape}")
