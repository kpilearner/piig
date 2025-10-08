"""
Stage 1: Pretrain Decomposition Network

Train the multi-task decomposition network with pseudo-labels generated
from RGB-Infrared pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion
from decomposition.losses import MultiTaskPretrainLoss, PerceptualLoss
from decomposition.pseudo_labels import PseudoLabelGenerator
from utils.data_utils import RGBInfraredDataset, create_dataloaders
from utils.visualization import visualize_decomposition, plot_training_curves


def train_one_epoch(
    model,
    fusion_module,
    dataloader,
    loss_fn,
    pseudo_gen,
    optimizer,
    device,
    epoch,
    writer=None,
    log_interval=50
):
    """Train for one epoch"""
    model.train()
    fusion_module.train()

    epoch_losses = {
        'total': [],
        'intensity': [],
        'material': [],
        'context': [],
        'fusion': []
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        infrared = batch['infrared'].to(device)
        image_ids = batch['image_id']

        # Generate pseudo-labels
        pseudo_intensity, pseudo_material, pseudo_context = pseudo_gen(
            rgb, infrared, image_ids
        )

        # Forward pass
        pred_intensity, pred_material_logits, pred_context = model(rgb)

        # Physics-inspired fusion
        fused_output = fusion_module(pred_intensity, pred_material_logits, pred_context)

        # Compute loss
        loss, loss_dict = loss_fn(
            pred_intensity,
            pred_material_logits,
            pred_context,
            fused_output,
            pseudo_intensity,
            pseudo_material,
            pseudo_context,
            infrared
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log losses
        for key in epoch_losses.keys():
            epoch_losses[key].append(loss_dict[key])

        # Update progress bar
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Log to tensorboard
        if writer and batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}_loss', value, global_step)

    # Average losses for epoch
    avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    return avg_losses


@torch.no_grad()
def validate(
    model,
    fusion_module,
    dataloader,
    loss_fn,
    pseudo_gen,
    device,
    epoch,
    writer=None,
    save_vis_path=None
):
    """Validate the model"""
    model.eval()
    fusion_module.eval()

    val_losses = {
        'total': [],
        'intensity': [],
        'material': [],
        'context': [],
        'fusion': []
    }

    # For visualization
    vis_batch = None

    for batch in tqdm(dataloader, desc="Validation"):
        rgb = batch['rgb'].to(device)
        infrared = batch['infrared'].to(device)
        image_ids = batch['image_id']

        # Generate pseudo-labels
        pseudo_intensity, pseudo_material, pseudo_context = pseudo_gen(
            rgb, infrared, image_ids
        )

        # Forward pass
        pred_intensity, pred_material_logits, pred_context = model(rgb)
        fused_output = fusion_module(pred_intensity, pred_material_logits, pred_context)

        # Compute loss
        loss, loss_dict = loss_fn(
            pred_intensity,
            pred_material_logits,
            pred_context,
            fused_output,
            pseudo_intensity,
            pseudo_material,
            pseudo_context,
            infrared
        )

        # Log losses
        for key in val_losses.keys():
            val_losses[key].append(loss_dict[key])

        # Save first batch for visualization
        if vis_batch is None:
            vis_batch = {
                'rgb': rgb,
                'intensity': pred_intensity,
                'material_logits': pred_material_logits,
                'context': pred_context,
                'infrared': infrared
            }

    # Average losses
    avg_losses = {k: sum(v) / len(v) for k, v in val_losses.items()}

    # Log to tensorboard
    if writer:
        for key, value in avg_losses.items():
            writer.add_scalar(f'val/{key}_loss', value, epoch)

    # Visualize
    if save_vis_path and vis_batch:
        visualize_decomposition(
            vis_batch['rgb'][:4],
            vis_batch['intensity'][:4],
            vis_batch['material_logits'][:4],
            vis_batch['context'][:4],
            vis_batch['infrared'][:4],
            save_path=save_vis_path,
            num_samples=4
        )

    return avg_losses


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_root=args.train_data,
        val_root=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    model = MultiTaskDecompositionNet(
        backbone=args.backbone,
        pretrained=True,
        num_material_classes=args.num_material_classes,
        context_channels=args.context_channels
    ).to(device)

    fusion_module = PhysicsInspiredFusion(
        num_material_classes=args.num_material_classes,
        context_channels=args.context_channels,
        hidden_dim=args.fusion_hidden_dim
    ).to(device)

    # Create loss function
    loss_fn = MultiTaskPretrainLoss(
        lambda_intensity=args.lambda_intensity,
        lambda_material=args.lambda_material,
        lambda_context=args.lambda_context,
        lambda_fusion=args.lambda_fusion
    ).to(device)

    # Create pseudo-label generator
    pseudo_gen = PseudoLabelGenerator(
        num_material_classes=args.num_material_classes,
        context_channels=args.context_channels,
        material_method=args.material_method,
        cache_materials=True
    )

    # Create optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(fusion_module.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )

    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_losses = train_one_epoch(
            model, fusion_module, train_loader, loss_fn, pseudo_gen,
            optimizer, device, epoch, writer, args.log_interval
        )

        print(f"Train Loss: {train_losses['total']:.4f}")

        # Validate
        if val_loader and (epoch + 1) % args.val_interval == 0:
            vis_path = os.path.join(args.vis_dir, f"epoch_{epoch+1}.png")
            val_losses = validate(
                model, fusion_module, val_loader, loss_fn, pseudo_gen,
                device, epoch, writer, vis_path
            )

            print(f"Val Loss: {val_losses['total']:.4f}")

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'fusion_state_dict': fusion_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total']
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'fusion_state_dict': fusion_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total']
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Step scheduler
        scheduler.step()

    print("\nTraining completed!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain Decomposition Network')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data root')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation data root')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for training')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--num_material_classes', type=int, default=32,
                       help='Number of material classes')
    parser.add_argument('--context_channels', type=int, default=8,
                       help='Number of context channels')
    parser.add_argument('--fusion_hidden_dim', type=int, default=64,
                       help='Hidden dimension for fusion module')

    # Loss arguments
    parser.add_argument('--lambda_intensity', type=float, default=1.0,
                       help='Weight for intensity loss')
    parser.add_argument('--lambda_material', type=float, default=1.0,
                       help='Weight for material loss')
    parser.add_argument('--lambda_context', type=float, default=0.5,
                       help='Weight for context loss')
    parser.add_argument('--lambda_fusion', type=float, default=2.0,
                       help='Weight for fusion reconstruction loss')

    # Pseudo-label arguments
    parser.add_argument('--material_method', type=str, default='simple',
                       choices=['kmeans', 'simple'],
                       help='Method for material pseudo-label generation')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/decomposition',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/decomposition',
                       help='Directory for tensorboard logs')
    parser.add_argument('--vis_dir', type=str, default='./visualizations/decomposition',
                       help='Directory for visualizations')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log interval (batches)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Checkpoint save interval (epochs)')

    # Config file (optional)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (overrides command line args)')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    main(args)
