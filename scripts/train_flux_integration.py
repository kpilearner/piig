"""
Stage 2: Train FLUX Integration

Integrate pretrained decomposition network with FLUX and train with LoRA.
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

from flux_integration.model import DecompositionGuidedFluxModel
from utils.data_utils import RGBInfraredMaskDataset
from utils.visualization import visualize_generation

# NOTE: This script provides a training template.
# The actual FLUX integration requires modifications based on the specific
# FLUX architecture and diffusion training pipeline.


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    writer=None,
    log_interval=50
):
    """
    Train for one epoch.

    NOTE: This is a placeholder. Actual implementation needs:
    1. Proper diffusion loss computation
    2. FLUX-specific training procedures
    3. Handling of timestep sampling
    4. Gradient accumulation if needed
    """
    model.train()

    epoch_loss = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        infrared = batch['infrared'].to(device)
        mask = batch['mask'].to(device) if 'mask' in batch else None

        # TODO: Implement actual FLUX training step
        # This requires:
        # 1. Sample timestep
        # 2. Add noise to infrared image
        # 3. Extract decomposition features
        # 4. Forward through FLUX with decomposition guidance
        # 5. Compute diffusion loss (MSE between predicted and true noise)

        # Placeholder loss computation
        optimizer.zero_grad()

        # Extract decomposition features
        decomp_features = model.encode_decomposition_features(rgb)

        # TODO: Implement FLUX forward pass with decomposition guidance
        # For now, just a placeholder
        loss = torch.tensor(0.0, requires_grad=True, device=device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log
        epoch_loss.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if writer and batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)

    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
    return avg_loss


@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    epoch,
    writer=None,
    save_vis_path=None
):
    """
    Validate the model.

    NOTE: This generates images and computes reconstruction metrics.
    """
    model.eval()

    val_losses = []
    vis_batch = None

    for batch in tqdm(dataloader, desc="Validation"):
        rgb = batch['rgb'].to(device)
        infrared = batch['infrared'].to(device)
        mask = batch['mask'].to(device) if 'mask' in batch else None

        # TODO: Implement actual FLUX generation
        # generated_ir = model(rgb, mask, num_inference_steps=50)

        # Placeholder: just use decomposition for now
        generated_ir = torch.zeros_like(infrared)

        # Compute reconstruction loss
        loss = nn.functional.mse_loss(generated_ir, infrared)
        val_losses.append(loss.item())

        # Save first batch for visualization
        if vis_batch is None:
            vis_batch = {
                'rgb': rgb,
                'generated_ir': generated_ir,
                'gt_infrared': infrared
            }

    avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

    # Log to tensorboard
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)

    # Visualize
    if save_vis_path and vis_batch:
        # Get decomposition for visualization
        decomposition = None
        try:
            intensity, material_logits, context = model.decomposition_net(vis_batch['rgb'][:4])
            decomposition = {
                'intensity': intensity,
                'material_logits': material_logits,
                'context': context
            }
        except:
            pass

        visualize_generation(
            vis_batch['rgb'][:4],
            vis_batch['generated_ir'][:4],
            vis_batch['gt_infrared'][:4],
            decomposition,
            save_path=save_vis_path,
            num_samples=4
        )

    return avg_loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Create dataset
    print("Loading datasets...")
    train_dataset = RGBInfraredMaskDataset(
        data_root=args.train_data,
        image_size=args.image_size,
        mode='train',
        random_mask_ratio=args.mask_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if args.val_data:
        val_dataset = RGBInfraredMaskDataset(
            data_root=args.val_data,
            image_size=args.image_size,
            mode='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    model = DecompositionGuidedFluxModel(
        flux_model_id=args.flux_model_id,
        decomposition_checkpoint=args.decomposition_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        freeze_decomposition=args.freeze_decomposition,
        num_material_classes=args.num_material_classes,
        context_channels=args.context_channels
    ).to(device)

    # Create optimizer (only for LoRA parameters + cross-attention)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )

    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    print("\nStarting training...")
    print("NOTE: This is a template. FLUX integration requires additional implementation.")

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, args.log_interval
        )

        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader and (epoch + 1) % args.val_interval == 0:
            vis_path = os.path.join(args.vis_dir, f"epoch_{epoch+1}.png")
            val_loss = validate(
                model, val_loader, device, epoch, writer, vis_path
            )

            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Step scheduler
        scheduler.step()

    print("\nTraining completed!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FLUX Integration')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data root')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation data root')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for training')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                       help='Ratio for random mask generation')

    # Model arguments
    parser.add_argument('--flux_model_id', type=str,
                       default='black-forest-labs/FLUX.1-Fill-dev',
                       help='FLUX model ID')
    parser.add_argument('--decomposition_checkpoint', type=str, required=True,
                       help='Path to pretrained decomposition checkpoint')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--freeze_decomposition', action='store_true',
                       help='Freeze decomposition network')
    parser.add_argument('--num_material_classes', type=int, default=32,
                       help='Number of material classes')
    parser.add_argument('--context_channels', type=int, default=8,
                       help='Number of context channels')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (smaller for FLUX)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')

    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/flux',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/flux',
                       help='Directory for tensorboard logs')
    parser.add_argument('--vis_dir', type=str, default='./visualizations/flux',
                       help='Directory for visualizations')
    parser.add_argument('--log_interval', type=int, default=20,
                       help='Log interval (batches)')
    parser.add_argument('--val_interval', type=int, default=2,
                       help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Checkpoint save interval (epochs)')

    # Config file (optional)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    main(args)
