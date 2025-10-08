"""
Multi-Task Pretraining Losses

Implements loss functions for training the decomposition network with pseudo-labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskPretrainLoss(nn.Module):
    """
    Combined loss for multi-task decomposition pretraining.

    Loss components:
    1. Intensity loss (MSE): Match pseudo temperature labels
    2. Material loss (Cross-Entropy): Match pseudo material labels
    3. Context loss (MSE): Match pseudo context labels
    4. Fusion reconstruction loss: Fused output should match GT infrared
    """

    def __init__(
        self,
        lambda_intensity=1.0,
        lambda_material=1.0,
        lambda_context=1.0,
        lambda_fusion=2.0
    ):
        """
        Args:
            lambda_intensity: Weight for intensity loss
            lambda_material: Weight for material loss
            lambda_context: Weight for context loss
            lambda_fusion: Weight for fusion reconstruction loss
        """
        super().__init__()

        self.lambda_intensity = lambda_intensity
        self.lambda_material = lambda_material
        self.lambda_context = lambda_context
        self.lambda_fusion = lambda_fusion

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for unknown regions

    def forward(
        self,
        pred_intensity,
        pred_material_logits,
        pred_context,
        fused_output,
        pseudo_intensity,
        pseudo_material,
        pseudo_context,
        gt_infrared
    ):
        """
        Args:
            pred_intensity: [B, 1, H, W] Predicted intensity
            pred_material_logits: [B, num_classes, H, W] Predicted material logits
            pred_context: [B, context_channels, H, W] Predicted context
            fused_output: [B, 1, H, W] Fused output from PhysicsInspiredFusion
            pseudo_intensity: [B, 1, H, W] Pseudo intensity label
            pseudo_material: [B, H, W] Pseudo material label (class indices)
            pseudo_context: [B, context_channels, H, W] Pseudo context label
            gt_infrared: [B, 1, H, W] Ground truth infrared image

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components for logging
        """
        # 1. Intensity loss (MSE)
        loss_intensity = self.mse_loss(pred_intensity, pseudo_intensity)

        # 2. Material loss (Cross-Entropy)
        loss_material = self.ce_loss(pred_material_logits, pseudo_material)

        # 3. Context loss (MSE)
        loss_context = self.mse_loss(pred_context, pseudo_context)

        # 4. Fusion reconstruction loss (MSE with GT infrared)
        loss_fusion = self.mse_loss(fused_output, gt_infrared)

        # Combined loss
        total_loss = (
            self.lambda_intensity * loss_intensity +
            self.lambda_material * loss_material +
            self.lambda_context * loss_context +
            self.lambda_fusion * loss_fusion
        )

        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'intensity': loss_intensity.item(),
            'material': loss_material.item(),
            'context': loss_context.item(),
            'fusion': loss_fusion.item()
        }

        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    Optional: Perceptual loss using VGG features for better visual quality.
    Can be added to fusion reconstruction loss.
    """

    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super().__init__()
        from torchvision.models import vgg16

        vgg = vgg16(pretrained=True).features.eval()
        self.layers = layers

        # Extract specific VGG layers
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(self.layer_name_mapping['relu1_2'] + 1):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(self.layer_name_mapping['relu1_2'] + 1, self.layer_name_mapping['relu2_2'] + 1):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(self.layer_name_mapping['relu2_2'] + 1, self.layer_name_mapping['relu3_3'] + 1):
            self.slice3.add_module(str(x), vgg[x])

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] Predicted image
            target: [B, 1, H, W] Target image

        Returns:
            loss: Perceptual loss
        """
        # Convert single channel to 3-channel for VGG
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

        # Extract features
        pred_relu1 = self.slice1(pred)
        pred_relu2 = self.slice2(pred_relu1)
        pred_relu3 = self.slice3(pred_relu2)

        target_relu1 = self.slice1(target)
        target_relu2 = self.slice2(target_relu1)
        target_relu3 = self.slice3(target_relu2)

        # Compute L1 loss on features
        loss = (
            F.l1_loss(pred_relu1, target_relu1) +
            F.l1_loss(pred_relu2, target_relu2) +
            F.l1_loss(pred_relu3, target_relu3)
        )

        return loss


class ConsistencyLoss(nn.Module):
    """
    Optional: Consistency loss to ensure decomposed components are meaningful.

    For example:
    - High intensity regions should correspond to specific material classes
    - Context should be spatially smooth
    """

    def __init__(self):
        super().__init__()

    def forward(self, intensity, material_logits, context):
        """
        Args:
            intensity: [B, 1, H, W]
            material_logits: [B, num_classes, H, W]
            context: [B, context_channels, H, W]

        Returns:
            loss: Consistency loss
        """
        # Spatial smoothness for context (Total Variation)
        diff_x = torch.abs(context[:, :, :, 1:] - context[:, :, :, :-1])
        diff_y = torch.abs(context[:, :, 1:, :] - context[:, :, :-1, :])
        tv_loss = torch.mean(diff_x) + torch.mean(diff_y)

        return tv_loss


if __name__ == '__main__':
    # Test the loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create loss function
    loss_fn = MultiTaskPretrainLoss(
        lambda_intensity=1.0,
        lambda_material=1.0,
        lambda_context=0.5,
        lambda_fusion=2.0
    )

    # Test inputs
    B, H, W = 4, 256, 256
    num_classes = 32
    context_channels = 8

    pred_intensity = torch.randn(B, 1, H, W).to(device)
    pred_material_logits = torch.randn(B, num_classes, H, W).to(device)
    pred_context = torch.randn(B, context_channels, H, W).to(device)
    fused_output = torch.randn(B, 1, H, W).to(device)

    pseudo_intensity = torch.randn(B, 1, H, W).to(device)
    pseudo_material = torch.randint(0, num_classes, (B, H, W)).to(device)
    pseudo_context = torch.randn(B, context_channels, H, W).to(device)
    gt_infrared = torch.randn(B, 1, H, W).to(device)

    # Compute loss
    total_loss, loss_dict = loss_fn(
        pred_intensity,
        pred_material_logits,
        pred_context,
        fused_output,
        pseudo_intensity,
        pseudo_material,
        pseudo_context,
        gt_infrared
    )

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
