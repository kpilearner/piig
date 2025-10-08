"""
Visualization utilities for decomposition results and generation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os


def visualize_decomposition(
    rgb,
    intensity,
    material_logits,
    context,
    gt_infrared=None,
    save_path=None,
    num_samples=4
):
    """
    Visualize decomposition results.

    Args:
        rgb: [B, 3, H, W] RGB images
        intensity: [B, 1, H, W] Intensity predictions
        material_logits: [B, num_classes, H, W] Material logits
        context: [B, context_channels, H, W] Context features
        gt_infrared: Optional [B, 1, H, W] Ground truth infrared
        save_path: Optional path to save visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, rgb.shape[0])

    # Material visualization (argmax over classes and convert to color)
    material_labels = torch.argmax(material_logits, dim=1)  # [B, H, W]

    # Context visualization (take first 3 channels as RGB)
    context_vis = context[:, :3, :, :]
    context_vis = (context_vis - context_vis.min()) / (context_vis.max() - context_vis.min() + 1e-8)

    # Denormalize RGB for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb.device)
    rgb_vis = rgb * std + mean
    rgb_vis = torch.clamp(rgb_vis, 0, 1)

    # Create figure
    num_cols = 5 if gt_infrared is not None else 4
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # RGB
        axes[i, 0].imshow(rgb_vis[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title('RGB Input')
        axes[i, 0].axis('off')

        # Intensity
        axes[i, 1].imshow(intensity[i, 0].cpu().numpy(), cmap='hot')
        axes[i, 1].set_title('Intensity\n(Temperature-like)')
        axes[i, 1].axis('off')

        # Material
        material_colored = material_labels[i].cpu().numpy()
        axes[i, 2].imshow(material_colored, cmap='tab20')
        axes[i, 2].set_title('Material\n(32 classes)')
        axes[i, 2].axis('off')

        # Context
        axes[i, 3].imshow(context_vis[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 3].set_title('Context\n(first 3 channels)')
        axes[i, 3].axis('off')

        # GT Infrared (if provided)
        if gt_infrared is not None:
            axes[i, 4].imshow(gt_infrared[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 4].set_title('GT Infrared')
            axes[i, 4].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()
    plt.close()


def visualize_generation(
    rgb,
    generated_ir,
    gt_infrared,
    decomposition_dict=None,
    save_path=None,
    num_samples=4
):
    """
    Visualize generation results with optional decomposition.

    Args:
        rgb: [B, 3, H, W] RGB images
        generated_ir: [B, 1, H, W] Generated infrared
        gt_infrared: [B, 1, H, W] Ground truth infrared
        decomposition_dict: Optional dict with 'intensity', 'material_logits', 'context'
        save_path: Optional path to save visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, rgb.shape[0])

    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb.device)
    rgb_vis = rgb * std + mean
    rgb_vis = torch.clamp(rgb_vis, 0, 1)

    # Compute error map
    error_map = torch.abs(generated_ir - gt_infrared)

    # Create figure
    if decomposition_dict is not None:
        num_cols = 6  # RGB, Generated, GT, Error, Intensity, Material
        material_labels = torch.argmax(decomposition_dict['material_logits'], dim=1)
    else:
        num_cols = 4  # RGB, Generated, GT, Error

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # RGB
        axes[i, 0].imshow(rgb_vis[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title('RGB Input')
        axes[i, 0].axis('off')

        # Generated IR
        axes[i, 1].imshow(generated_ir[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Generated IR')
        axes[i, 1].axis('off')

        # GT IR
        axes[i, 2].imshow(gt_infrared[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title('GT Infrared')
        axes[i, 2].axis('off')

        # Error map
        im = axes[i, 3].imshow(error_map[i, 0].cpu().numpy(), cmap='hot')
        axes[i, 3].set_title('Error Map')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

        # Decomposition (if provided)
        if decomposition_dict is not None:
            # Intensity
            axes[i, 4].imshow(decomposition_dict['intensity'][i, 0].cpu().numpy(), cmap='hot')
            axes[i, 4].set_title('Intensity')
            axes[i, 4].axis('off')

            # Material
            axes[i, 5].imshow(material_labels[i].cpu().numpy(), cmap='tab20')
            axes[i, 5].set_title('Material')
            axes[i, 5].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()
    plt.close()


def plot_training_curves(
    loss_history,
    save_path=None
):
    """
    Plot training loss curves.

    Args:
        loss_history: Dict with keys like 'total', 'intensity', 'material', 'context', 'fusion'
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    loss_keys = list(loss_history.keys())

    for idx, key in enumerate(loss_keys[:6]):  # Plot up to 6 loss components
        axes[idx].plot(loss_history[key])
        axes[idx].set_title(f'{key.capitalize()} Loss')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')
        axes[idx].grid(True)

    # Hide unused subplots
    for idx in range(len(loss_keys), 6):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()
    plt.close()


def create_comparison_grid(
    images_dict,
    titles=None,
    save_path=None,
    nrow=8
):
    """
    Create a grid comparison of multiple image sets.

    Args:
        images_dict: Dict of {name: [B, C, H, W] tensor}
        titles: Optional list of titles for each row
        save_path: Optional path to save grid
        nrow: Number of images per row
    """
    fig, axes = plt.subplots(len(images_dict), 1, figsize=(20, len(images_dict) * 3))

    if len(images_dict) == 1:
        axes = [axes]

    for idx, (name, images) in enumerate(images_dict.items()):
        # Create grid
        if images.shape[1] == 1:  # Grayscale
            grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
            grid_np = grid[0].cpu().numpy()
            axes[idx].imshow(grid_np, cmap='gray')
        else:  # RGB
            grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
            grid_np = grid.cpu().permute(1, 2, 0).numpy()
            axes[idx].imshow(grid_np)

        title = titles[idx] if titles and idx < len(titles) else name
        axes[idx].set_title(title, fontsize=14)
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")

    plt.show()
    plt.close()


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization functions...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    B, H, W = 4, 256, 256
    rgb = torch.rand(B, 3, H, W).to(device)
    intensity = torch.rand(B, 1, H, W).to(device)
    material_logits = torch.randn(B, 32, H, W).to(device)
    context = torch.randn(B, 8, H, W).to(device)
    gt_infrared = torch.rand(B, 1, H, W).to(device)

    # Test decomposition visualization
    print("\nVisualizing decomposition...")
    visualize_decomposition(
        rgb, intensity, material_logits, context, gt_infrared,
        num_samples=2
    )

    # Test generation visualization
    print("\nVisualizing generation...")
    generated_ir = torch.rand(B, 1, H, W).to(device)
    decomposition_dict = {
        'intensity': intensity,
        'material_logits': material_logits,
        'context': context
    }
    visualize_generation(
        rgb, generated_ir, gt_infrared, decomposition_dict,
        num_samples=2
    )

    print("\nVisualization tests successful!")
