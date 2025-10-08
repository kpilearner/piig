"""
Pseudo-Label Generation Utilities

Generate pseudo-labels from RGB-Infrared pairs for decomposition network pretraining.
These are NOT true physical quantities but task-relevant approximations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def generate_pseudo_intensity(infrared_image, normalize=True):
    """
    Generate pseudo intensity labels from infrared images.

    Strategy: Use infrared intensity as proxy for "temperature-like" distribution.
    Assumption: Brighter regions in infrared ≈ hotter objects (Stefan-Boltzmann: I ∝ T^4)

    Args:
        infrared_image: [B, 1, H, W] Infrared image in [0, 1] range
        normalize: Whether to normalize to [0, 1] range

    Returns:
        pseudo_intensity: [B, 1, H, W] Pseudo intensity labels
    """
    # Simple version: directly use infrared intensity
    pseudo_intensity = infrared_image.clone()

    if normalize:
        # Normalize per image to [0, 1] range
        B = infrared_image.shape[0]
        for i in range(B):
            min_val = infrared_image[i].min()
            max_val = infrared_image[i].max()
            if max_val > min_val:
                pseudo_intensity[i] = (infrared_image[i] - min_val) / (max_val - min_val)

    return pseudo_intensity


def generate_pseudo_material_from_panoptic(semantic_image, num_classes=32):
    """
    从panoptic分割图直接生成材料伪标签（推荐）

    Strategy: 使用panoptic_img中的颜色块作为材料类别
    每个唯一的RGB颜色 = 一个对象实例 → 映射到材料类别

    Args:
        semantic_image: [B, 3, H, W] Panoptic分割图（RGB颜色块）
        num_classes: 最大材料类别数

    Returns:
        pseudo_material: [B, H, W] Pseudo material labels (class indices)
    """
    B, _, H, W = semantic_image.shape
    device = semantic_image.device

    pseudo_material = torch.zeros(B, H, W, dtype=torch.long, device=device)

    for i in range(B):
        # 将RGB颜色转换为单一标签
        rgb = semantic_image[i].permute(1, 2, 0)  # [H, W, 3]
        rgb_int = (rgb * 255).long()  # 转换为整数

        # 将RGB编码为单一整数: R * 256^2 + G * 256 + B
        color_ids = rgb_int[:, :, 0] * 65536 + rgb_int[:, :, 1] * 256 + rgb_int[:, :, 2]

        # 获取唯一颜色
        unique_colors = torch.unique(color_ids)

        # 映射到类别索引（限制在num_classes内）
        color_to_class = {}
        for idx, color in enumerate(unique_colors):
            color_to_class[color.item()] = idx % num_classes

        # 生成材料标签
        material_labels = torch.zeros(H, W, dtype=torch.long, device=device)
        for color, class_id in color_to_class.items():
            mask = (color_ids == color)
            material_labels[mask] = class_id

        pseudo_material[i] = material_labels

    return pseudo_material


def generate_pseudo_material(rgb_image, infrared_image, num_classes=32, method='kmeans'):
    """
    Generate pseudo material labels using clustering on RGB-IR features.

    ⚠️ 注意：如果你有panoptic_img，推荐使用 generate_pseudo_material_from_panoptic()
    这个函数用于没有语义分割图的情况

    Strategy: Cluster image regions based on RGB+IR features.
    Assumption: Similar appearance + thermal behavior ≈ similar material.

    Args:
        rgb_image: [B, 3, H, W] RGB image in [0, 1] range
        infrared_image: [B, 1, H, W] Infrared image in [0, 1] range
        num_classes: Number of material classes
        method: Clustering method ('kmeans' or 'simple')

    Returns:
        pseudo_material: [B, H, W] Pseudo material labels (class indices)
    """
    B, _, H, W = rgb_image.shape
    device = rgb_image.device

    pseudo_material = torch.zeros(B, H, W, dtype=torch.long, device=device)

    for i in range(B):
        # Combine RGB and IR features
        rgb_i = rgb_image[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        ir_i = infrared_image[i, 0].cpu().numpy()  # [H, W]

        # Stack features: [H, W, 4] (RGB + IR)
        features = np.concatenate([rgb_i, ir_i[..., None]], axis=-1)
        features_flat = features.reshape(-1, 4)  # [H*W, 4]

        if method == 'kmeans':
            # K-means clustering
            kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_flat)
            labels = labels.reshape(H, W)

        elif method == 'simple':
            # Simple quantization (faster but less accurate)
            # Quantize RGB to bins
            rgb_bins = (rgb_i * 4).astype(int).clip(0, 3)  # 4 bins per channel
            ir_bins = (ir_i * 4).astype(int).clip(0, 3)    # 4 bins for IR

            # Combine into single label (4^3 * 4 = 256 possible combinations)
            labels = (
                rgb_bins[..., 0] * 64 +
                rgb_bins[..., 1] * 16 +
                rgb_bins[..., 2] * 4 +
                ir_bins
            )
            # Map to num_classes
            labels = labels % num_classes

        else:
            raise ValueError(f"Unknown method: {method}")

        pseudo_material[i] = torch.from_numpy(labels).long().to(device)

    return pseudo_material


def generate_pseudo_context(rgb_image, infrared_image, context_channels=8):
    """
    Generate pseudo context labels representing environmental factors.

    Strategy: Extract multi-scale spatial features from RGB+IR.
    These represent global context like scene type, lighting, etc.

    Args:
        rgb_image: [B, 3, H, W] RGB image
        infrared_image: [B, 1, H, W] Infrared image
        context_channels: Number of context channels

    Returns:
        pseudo_context: [B, context_channels, H, W] Pseudo context features
    """
    B, _, H, W = rgb_image.shape
    device = rgb_image.device

    # Combine RGB and IR
    combined = torch.cat([rgb_image, infrared_image], dim=1)  # [B, 4, H, W]

    # Multi-scale pooling to get global context
    pool_sizes = [2, 4, 8, 16]
    context_features = []

    for pool_size in pool_sizes[:context_channels // 2]:
        pooled = F.adaptive_avg_pool2d(combined, pool_size)
        upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
        context_features.append(upsampled)

    # Gradient-based features (edge detection)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Apply Sobel to grayscale (average of RGB)
    gray = rgb_image.mean(dim=1, keepdim=True)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    context_features.append(grad_x)
    context_features.append(grad_y)

    # Pad if needed
    while len(context_features) < context_channels:
        # Add noise as placeholder for remaining channels
        noise = torch.randn(B, 1, H, W, device=device) * 0.1
        context_features.append(noise)

    # Concatenate and take first context_channels
    pseudo_context = torch.cat(context_features, dim=1)[:, :context_channels]

    # Normalize to [-1, 1] range
    pseudo_context = torch.tanh(pseudo_context)

    return pseudo_context


class PseudoLabelGenerator:
    """
    Wrapper class for generating all pseudo-labels in one call.
    """

    def __init__(
        self,
        num_material_classes=32,
        context_channels=8,
        material_method='kmeans',
        cache_materials=True
    ):
        """
        Args:
            num_material_classes: Number of material classes
            context_channels: Number of context channels
            material_method: Method for material clustering
            cache_materials: Whether to cache material labels (slower but more consistent)
        """
        self.num_material_classes = num_material_classes
        self.context_channels = context_channels
        self.material_method = material_method
        self.cache_materials = cache_materials

        # Cache for material labels (if enabled)
        self.material_cache = {}

    def __call__(self, rgb_image, infrared_image, image_ids=None):
        """
        Generate all pseudo-labels for a batch.

        Args:
            rgb_image: [B, 3, H, W] RGB images
            infrared_image: [B, 1, H, W] Infrared images
            image_ids: Optional list of image IDs for caching

        Returns:
            pseudo_intensity: [B, 1, H, W]
            pseudo_material: [B, H, W]
            pseudo_context: [B, context_channels, H, W]
        """
        # Generate intensity labels
        pseudo_intensity = generate_pseudo_intensity(infrared_image, normalize=True)

        # Generate material labels (with optional caching)
        if self.cache_materials and image_ids is not None:
            B = rgb_image.shape[0]
            device = rgb_image.device
            H, W = infrared_image.shape[2:]

            pseudo_material = torch.zeros(B, H, W, dtype=torch.long, device=device)

            for i in range(B):
                img_id = image_ids[i]
                if img_id in self.material_cache:
                    pseudo_material[i] = self.material_cache[img_id]
                else:
                    material_i = generate_pseudo_material(
                        rgb_image[i:i+1],
                        infrared_image[i:i+1],
                        num_classes=self.num_material_classes,
                        method=self.material_method
                    )
                    pseudo_material[i] = material_i[0]
                    self.material_cache[img_id] = material_i[0]
        else:
            pseudo_material = generate_pseudo_material(
                rgb_image,
                infrared_image,
                num_classes=self.num_material_classes,
                method=self.material_method
            )

        # Generate context labels
        pseudo_context = generate_pseudo_context(
            rgb_image,
            infrared_image,
            context_channels=self.context_channels
        )

        return pseudo_intensity, pseudo_material, pseudo_context


if __name__ == '__main__':
    # Test pseudo-label generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    B, H, W = 2, 256, 256
    rgb = torch.rand(B, 3, H, W).to(device)
    infrared = torch.rand(B, 1, H, W).to(device)

    # Generate pseudo-labels
    print("Generating pseudo-labels...")

    pseudo_gen = PseudoLabelGenerator(
        num_material_classes=32,
        context_channels=8,
        material_method='simple'  # Use 'simple' for faster testing
    )

    intensity, material, context = pseudo_gen(rgb, infrared)

    print(f"Pseudo intensity shape: {intensity.shape}, range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    print(f"Pseudo material shape: {material.shape}, unique classes: {material.unique().numel()}")
    print(f"Pseudo context shape: {context.shape}, range: [{context.min():.3f}, {context.max():.3f}]")

    print("\nPseudo-label generation successful!")
