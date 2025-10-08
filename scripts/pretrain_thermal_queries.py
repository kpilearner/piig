"""
Stage 1: Thermal Query Decoder Pretraining

Train the ThermalQueryDecoder to learn thermal patterns from RGB-IR pairs.
This is much simpler than the multi-task decomposition approach!

Key improvements:
- No pseudo-labels needed
- Single MSE loss (very stable)
- End-to-end learning of thermal patterns
- Queries implicitly learn material/context information

Usage:
    python scripts/pretrain_thermal_queries.py \\
        --config configs/pretrain_thermal_queries.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decomposition.thermal_query_decoder import ThermalQueryDecoder
from utils.parquet_dataloader import create_dataloaders


class ThermalQueryTrainer:
    """Trainer for Thermal Query Decoder"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directories
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        # TensorBoard
        tb_dir = config.get('tensorboard_dir', str(self.save_dir / 'logs'))
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Create model
        print("\n" + "=" * 70)
        print("🏗️  Creating Thermal Query Decoder...")
        print("=" * 70)

        self.model = ThermalQueryDecoder(
            backbone=config['model']['backbone'],
            pretrained=config['model']['pretrained'],
            num_thermal_queries=config['model']['num_thermal_queries'],
            hidden_dim=config['model']['hidden_dim'],
            num_scales=config['model']['num_scales'],
            dec_layers=config['model']['dec_layers'],
            nheads=config['model']['nheads'],
            dim_feedforward=config['model']['dim_feedforward'],
            pre_norm=config['model']['pre_norm'],
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")

        # Loss function
        if config['loss']['type'] == 'MSE':
            self.criterion = nn.MSELoss()
        elif config['loss']['type'] == 'L1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {config['loss']['type']}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # Learning rate scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs'],
                eta_min=config['training']['learning_rate'] * 0.01,
            )
        else:
            self.scheduler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Get data
            rgb = batch['rgb'].to(self.device)
            infrared = batch['infrared'].to(self.device)

            # Forward pass
            pred_infrared = self.model(rgb)

            # Compute loss
            # Target: grayscale infrared
            target_ir = infrared.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            loss = self.criterion(pred_infrared, target_ir)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )

            self.optimizer.step()

            # Record loss
            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # TensorBoard logging
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Average loss
        epoch_loss /= len(train_loader)

        return epoch_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0

        for batch in tqdm(val_loader, desc="Validation"):
            rgb = batch['rgb'].to(self.device)
            infrared = batch['infrared'].to(self.device)

            # Forward pass
            pred_infrared = self.model(rgb)

            # Compute loss
            target_ir = infrared.mean(dim=1, keepdim=True)
            loss = self.criterion(pred_infrared, target_ir)

            val_loss += loss.item()

        # Average loss
        val_loss /= len(val_loader)

        return val_loss

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f"   💾 Checkpoint saved: {filepath}")

        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   ⭐ Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"✅ Checkpoint loaded! Resuming from epoch {self.epoch}")

    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "=" * 70)
        print("🚀 Starting Thermal Query Decoder Training")
        print("=" * 70)

        num_epochs = self.config['training']['num_epochs']

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss = self.validate(val_loader)

                # Print results
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")

                # TensorBoard
                self.writer.add_scalar('val/loss', val_loss, epoch)

                # Save best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save checkpoint
                if (epoch + 1) % self.config['training']['save_interval'] == 0:
                    self.save_checkpoint(f'epoch_{epoch}.pth', is_best=is_best)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

        # Save final model
        self.save_checkpoint('final_model.pth')

        print("\n" + "=" * 70)
        print("✅ Training complete!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        print(f"   Model saved to: {self.save_dir}")
        print("=" * 70)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Pretrain Thermal Query Decoder')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--parquet', type=str, default=None,
                        help='Override parquet path')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override number of epochs')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.parquet:
        config['data']['parquet_path'] = args.parquet
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs

    # Print config
    print("=" * 70)
    print("Configuration:")
    print("=" * 70)
    print(f"  Data: {config['data']['parquet_path']}")
    print(f"  Backbone: {config['model']['backbone']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Thermal queries: {config['model']['num_thermal_queries']}")
    print(f"  Decoder layers: {config['model']['dec_layers']}")
    print(f"  Save dir: {config['save_dir']}")
    print("=" * 70)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        parquet_path=config['data']['parquet_path'],
        split_file=config['data']['split_file'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        normalize=config['data']['normalize'],
    )

    # Create trainer
    trainer = ThermalQueryTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
