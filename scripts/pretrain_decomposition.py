"""
阶段1: 分解网络预训练

训练多任务分解网络学习物理启发的表示：
- Intensity: 类似温度的亮度分布
- Material: 类似发射率的材料类别（从panoptic_img生成）
- Context: 环境特征

使用与ICEdit_contrastive完全相同的数据格式（parquet）
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

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion
from decomposition.losses import MultiTaskPretrainLoss
from decomposition.pseudo_labels import (
    generate_pseudo_intensity,
    generate_pseudo_material_from_panoptic,
    generate_pseudo_context
)
from utils.parquet_dataloader import create_dataloaders


class DecompositionTrainer:
    """分解网络训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))

        # 创建模型
        print("🏗️  创建分解网络...")
        self.model = MultiTaskDecompositionNet(
            backbone=config['model']['backbone'],
            pretrained=True,
            num_material_classes=config['model']['num_material_classes'],
            context_channels=config['model']['context_channels'],
        ).to(self.device)

        self.fusion_module = PhysicsInspiredFusion(
            num_material_classes=config['model']['num_material_classes'],
            context_channels=config['model']['context_channels'],
            hidden_dim=64,
        ).to(self.device)

        print(f"   参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        # 创建损失函数
        self.criterion = MultiTaskPretrainLoss(
            lambda_intensity=config['loss']['lambda_intensity'],
            lambda_material=config['loss']['lambda_material'],
            lambda_context=config['loss']['lambda_context'],
            lambda_fusion=config['loss']['lambda_fusion'],
        )

        # 创建优化器（包含model和fusion_module）
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.fusion_module.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['learning_rate'] * 0.01,
        )

        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0,
            'intensity': 0,
            'material': 0,
            'context': 0,
            'fusion': 0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到GPU
            rgb = batch['rgb'].to(self.device)
            infrared = batch['infrared'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            # 生成伪标签
            with torch.no_grad():
                # Intensity: 从红外图像生成
                ir_gray = infrared.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                pseudo_intensity = generate_pseudo_intensity(ir_gray, normalize=True)

                # Material: 从panoptic分割图生成（推荐方法）
                pseudo_material = generate_pseudo_material_from_panoptic(
                    semantic,
                    num_classes=self.config['model']['num_material_classes']
                )

                # Context: 从RGB-IR差异生成
                pseudo_context = generate_pseudo_context(rgb, ir_gray)

                # 目标红外图（用于fusion重建）
                target_ir = ir_gray

            # 前向传播
            pred_intensity, pred_material_logits, pred_context = self.model(rgb)

            # Physics-inspired fusion
            fused_output = self.fusion_module(pred_intensity, pred_material_logits, pred_context)

            # 计算损失
            loss, losses = self.criterion(
                pred_intensity,
                pred_material_logits,
                pred_context,
                fused_output,
                pseudo_intensity,
                pseudo_material,
                pseudo_context,
                target_ir
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（包含model和fusion_module）
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.fusion_module.parameters()),
                self.config['training']['grad_clip']
            )

            self.optimizer.step()

            # 记录损失
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key]

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'fusion': f"{losses['fusion']:.4f}",
            })

            # TensorBoard记录
            if self.global_step % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}_loss', value, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # 计算平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= len(train_loader)

        return epoch_losses

    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        self.fusion_module.eval()
        val_losses = {
            'total': 0,
            'intensity': 0,
            'material': 0,
            'context': 0,
            'fusion': 0,
        }

        for batch in tqdm(val_loader, desc="Validation"):
            rgb = batch['rgb'].to(self.device)
            infrared = batch['infrared'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            # 生成伪标签
            ir_gray = infrared.mean(dim=1, keepdim=True)
            pseudo_intensity = generate_pseudo_intensity(ir_gray, normalize=True)
            pseudo_material = generate_pseudo_material_from_panoptic(
                semantic,
                num_classes=self.config['model']['num_material_classes']
            )
            pseudo_context = generate_pseudo_context(rgb, ir_gray)
            target_ir = ir_gray

            # 前向传播
            pred_intensity, pred_material_logits, pred_context = self.model(rgb)
            fused_output = self.fusion_module(pred_intensity, pred_material_logits, pred_context)

            # 计算损失
            loss, losses = self.criterion(
                pred_intensity,
                pred_material_logits,
                pred_context,
                fused_output,
                pseudo_intensity,
                pseudo_material,
                pseudo_context,
                target_ir
            )

            # 累积损失
            for key in val_losses.keys():
                val_losses[key] += losses[key]

        # 计算平均损失
        for key in val_losses.keys():
            val_losses[key] /= len(val_loader)

        return val_losses

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'fusion_state_dict': self.fusion_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        print(f"   💾 检查点已保存: {filepath}")

        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   ⭐ 最佳模型已保存: {best_path}")

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("开始训练分解网络")
        print("=" * 70)

        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_losses = self.train_epoch(train_loader)

            # 验证
            val_losses = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 打印结果
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Int: {train_losses['intensity']:.4f}, "
                  f"Mat: {train_losses['material']:.4f}, "
                  f"Ctx: {train_losses['context']:.4f}, "
                  f"Fus: {train_losses['fusion']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(Int: {val_losses['intensity']:.4f}, "
                  f"Mat: {val_losses['material']:.4f}, "
                  f"Ctx: {val_losses['context']:.4f}, "
                  f"Fus: {val_losses['fusion']:.4f})")

            # TensorBoard记录
            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}_loss', value, epoch)

            # 保存检查点
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth', is_best=is_best)

        # 保存最终模型
        self.save_checkpoint('final_model.pth')

        print("\n" + "=" * 70)
        print("✅ 训练完成!")
        print(f"   最佳验证损失: {self.best_val_loss:.4f}")
        print(f"   模型保存路径: {self.save_dir}")
        print("=" * 70)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='预训练分解网络')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--parquet', type=str, default=None,
                       help='Parquet文件路径（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖
    if args.parquet:
        config['data']['parquet_path'] = args.parquet
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs

    print("=" * 70)
    print("配置信息:")
    print("=" * 70)
    print(f"  数据路径: {config['data']['parquet_path']}")
    print(f"  批次大小: {config['data']['batch_size']}")
    print(f"  训练轮数: {config['training']['num_epochs']}")
    print(f"  学习率: {config['training']['learning_rate']}")
    print(f"  保存路径: {config['save_dir']}")
    print("=" * 70)

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        parquet_path=config['data']['parquet_path'],
        split_file=config['data']['split_file'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        normalize=True,
    )

    # 创建训练器
    trainer = DecompositionTrainer(config)

    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
