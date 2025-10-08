# 方案D' 完整实现指南

## 整体训练流程

```
阶段1: 预训练多任务分解网络 (2-3周)
   RGB → [Decomposition Net] → {intensity, material, context}
   训练目标: 学习有意义的中间表示

阶段2: 联合训练FLUX生成 (2-3周)
   RGB → [Decomposition Net] → {intensity, material, context}
                                         ↓
                                   [FLUX Generator]
                                         ↓
                                    Infrared Image
   训练目标: 端到端优化生成质量

阶段3: 微调和优化 (1-2周)
   调整权重、消融实验、可视化分析
```

---

## 阶段1: 预训练多任务分解网络

### 1.1 为什么需要预训练？

**问题**: 如果直接端到端训练FLUX + 分解网络
```
RGB → [Decomposition] → {intensity, material, context}
              ↓
         [FLUX] → Infrared

问题:
1. 分解网络随机初始化，输出是噪声
2. FLUX无法从噪声中学到有用信息
3. 训练不稳定，很难收敛
```

**解决**: 先预训练分解网络
```
让分解网络先学会输出有意义的表示
然后再接入FLUX
```

### 1.2 预训练架构

**文件**: `train/src/decomposition/model.py`

```python
"""
多任务分解网络预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskDecompositionNet(nn.Module):
    """
    多任务分解网络

    输入: RGB图像 [B, 3, H, W]
    输出:
        - intensity_map: [B, 1, H, W] 亮度图
        - material_logits: [B, 32, H, W] 材质分类
        - context_features: [B, 8, H, W] 上下文特征
    """

    def __init__(
        self,
        backbone='resnet50',
        num_materials=32,
        num_context_channels=8,
        pretrained=True
    ):
        super().__init__()

        # 共享编码器
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            encoder_dim = 2048
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            encoder_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 上采样模块（用于恢复空间分辨率）
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(encoder_dim, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 任务头1: Intensity (亮度预测)
        self.intensity_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

        # 任务头2: Material (材质分类)
        self.material_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_materials, 1)  # 32类材质
        )

        # 任务头3: Context (上下文特征)
        self.context_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_context_channels, 1)
        )

        # 物理启发的融合模块（用于自监督）
        self.fusion = PhysicsInspiredFusion(num_materials)

        print(f"[INFO] MultiTaskDecompositionNet initialized")
        print(f"  - Backbone: {backbone}")
        print(f"  - Num materials: {num_materials}")
        print(f"  - Context channels: {num_context_channels}")

    def forward(self, rgb, return_fusion=False):
        """
        Args:
            rgb: [B, 3, H, W]
            return_fusion: 是否返回融合的红外预测

        Returns:
            dict with keys:
                - intensity: [B, 1, H, W]
                - material_logits: [B, 32, H, W]
                - context: [B, 8, H, W]
                - infrared_pred: [B, 3, H, W] (if return_fusion=True)
        """
        # 编码
        feat = self.encoder(rgb)  # [B, 2048, H/32, W/32]

        # 上采样
        feat_up = self.upsample(feat)  # [B, 128, H/4, W/4]

        # 最终上采样到原始分辨率
        B, C, H, W = rgb.shape
        feat_up = F.interpolate(feat_up, size=(H, W), mode='bilinear', align_corners=True)

        # 三个任务头
        intensity = self.intensity_head(feat_up)  # [B, 1, H, W]
        material_logits = self.material_head(feat_up)  # [B, 32, H, W]
        context = self.context_head(feat_up)  # [B, 8, H, W]

        output = {
            'intensity': intensity,
            'material_logits': material_logits,
            'context': context
        }

        # 如果需要，通过融合模块生成红外预测（用于自监督）
        if return_fusion:
            infrared_pred = self.fusion(intensity, material_logits, context)
            output['infrared_pred'] = infrared_pred

        return output


class PhysicsInspiredFusion(nn.Module):
    """
    物理启发的融合模块

    将intensity, material, context融合为红外图像
    """

    def __init__(self, num_materials=32):
        super().__init__()

        # 可学习的幂指数（启发自T^4）
        self.power = nn.Parameter(torch.tensor(2.5))

        # 材质到权重的映射
        self.material_modulator = nn.Sequential(
            nn.Conv2d(num_materials, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1]，类似发射率
        )

        # 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1)
        )

        # 组合权重
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, intensity, material_logits, context):
        """
        Args:
            intensity: [B, 1, H, W]
            material_logits: [B, 32, H, W]
            context: [B, 8, H, W]

        Returns:
            infrared_pred: [B, 3, H, W]
        """
        # Step 1: 基础辐射 (类似 T^power)
        base = torch.pow(intensity.clamp(min=1e-6), self.power)

        # Step 2: 材质调制 (类似发射率ε)
        material_weight = self.material_modulator(material_logits)
        modulated = base * material_weight

        # Step 3: 上下文贡献 (类似环境辐射)
        context_contrib = self.context_fusion(context)

        # Step 4: 组合
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        infrared_pred = (
            alpha * modulated.repeat(1, 3, 1, 1) +
            beta * context_contrib
        )

        return infrared_pred
```

### 1.3 伪标签生成

**文件**: `train/src/decomposition/pseudo_labels.py`

```python
"""
为多任务学习生成伪标签
"""

import torch
import torch.nn.functional as F


def generate_intensity_pseudo_label(infrared_gt):
    """
    从红外图生成intensity伪标签

    物理依据: I ∝ T^4 → T ∝ I^(1/4)

    Args:
        infrared_gt: [B, 3, H, W] 真实红外图

    Returns:
        intensity_pseudo: [B, 1, H, W] 归一化的亮度图
    """
    # 转为灰度
    gray = infrared_gt.mean(dim=1, keepdim=True)

    # 每个样本独立归一化
    B = gray.shape[0]
    intensity_pseudo = []

    for i in range(B):
        sample = gray[i]
        # 归一化到 [0, 1]
        sample_min = sample.min()
        sample_max = sample.max()

        normalized = (sample - sample_min) / (sample_max - sample_min + 1e-8)
        intensity_pseudo.append(normalized)

    intensity_pseudo = torch.stack(intensity_pseudo, dim=0)

    return intensity_pseudo


def generate_material_pseudo_label(semantic_map):
    """
    从语义分割生成material伪标签

    映射规则:
    - 将150类语义映射到32类材质

    Args:
        semantic_map: [B, 3, H, W] 语义分割图（RGB格式）

    Returns:
        material_pseudo: [B, H, W] 材质类别 (0-31)
    """
    # 将RGB语义图转为类别ID
    # 这里需要根据你的语义分割格式调整
    # 假设已经有一个函数 rgb_to_class_id

    # 简化版本：基于颜色聚类
    B, C, H, W = semantic_map.shape
    semantic_gray = semantic_map.mean(dim=1)  # [B, H, W]

    # 量化到32个level
    material_pseudo = (semantic_gray * 31).long().clamp(0, 31)

    return material_pseudo


def generate_context_pseudo_label(rgb, semantic_map=None):
    """
    生成上下文特征伪标签

    包括:
    - 天空检测
    - 阴影检测
    - 亮度分布
    - 等等

    Args:
        rgb: [B, 3, H, W]
        semantic_map: [B, 3, H, W] (可选)

    Returns:
        context_pseudo: [B, 8, H, W]
    """
    B, C, H, W = rgb.shape
    context_channels = []

    # 通道1: 整体亮度
    brightness = rgb.mean(dim=1, keepdim=True)
    context_channels.append(brightness)

    # 通道2: 对比度
    contrast = rgb.std(dim=1, keepdim=True)
    context_channels.append(contrast)

    # 通道3-4: 颜色信息
    context_channels.append(rgb[:, 0:1])  # R通道
    context_channels.append(rgb[:, 1:2])  # G通道

    # 通道5: 阴影检测（简化：亮度<阈值）
    shadow_mask = (brightness < 0.3).float()
    context_channels.append(shadow_mask)

    # 通道6-8: 梯度信息
    grad_x = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1])
    grad_x = F.pad(grad_x, (0, 1, 0, 0))  # 填充到原始尺寸
    grad_x = grad_x.mean(dim=1, keepdim=True)
    context_channels.append(grad_x)

    grad_y = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :])
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    grad_y = grad_y.mean(dim=1, keepdim=True)
    context_channels.append(grad_y)

    # 通道8: 纹理
    texture = (grad_x + grad_y) / 2
    context_channels.append(texture)

    # 拼接
    context_pseudo = torch.cat(context_channels, dim=1)

    return context_pseudo


def prepare_pseudo_labels(batch):
    """
    为一个batch准备所有伪标签

    Args:
        batch: dict with keys 'visible', 'infrared', 'semantic'

    Returns:
        pseudo_labels: dict with keys 'intensity', 'material', 'context'
    """
    visible = batch['visible']
    infrared = batch['infrared']
    semantic = batch.get('semantic', None)

    # 生成伪标签
    intensity_pseudo = generate_intensity_pseudo_label(infrared)
    material_pseudo = generate_material_pseudo_label(semantic if semantic is not None else visible)
    context_pseudo = generate_context_pseudo_label(visible, semantic)

    return {
        'intensity': intensity_pseudo,
        'material': material_pseudo,
        'context': context_pseudo
    }
```

### 1.4 预训练损失函数

**文件**: `train/src/decomposition/losses.py`

```python
"""
多任务预训练损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskPretrainLoss(nn.Module):
    """
    多任务预训练损失

    包括:
    1. Intensity预测损失
    2. Material分类损失
    3. Context重建损失
    4. 物理一致性损失（自监督）
    """

    def __init__(
        self,
        lambda_intensity=1.0,
        lambda_material=0.5,
        lambda_context=0.3,
        lambda_physics=0.5,
        lambda_infrared_recon=0.2
    ):
        super().__init__()

        self.lambda_intensity = lambda_intensity
        self.lambda_material = lambda_material
        self.lambda_context = lambda_context
        self.lambda_physics = lambda_physics
        self.lambda_infrared_recon = lambda_infrared_recon

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, pseudo_labels, infrared_gt):
        """
        Args:
            pred: dict from model output
                - intensity: [B, 1, H, W]
                - material_logits: [B, 32, H, W]
                - context: [B, 8, H, W]
                - infrared_pred: [B, 3, H, W] (如果有)
            pseudo_labels: dict
                - intensity: [B, 1, H, W]
                - material: [B, H, W]
                - context: [B, 8, H, W]
            infrared_gt: [B, 3, H, W]

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        # 1. Intensity损失
        loss_intensity = self.mse_loss(
            pred['intensity'],
            pseudo_labels['intensity']
        )

        # 2. Material分类损失
        loss_material = self.ce_loss(
            pred['material_logits'],
            pseudo_labels['material']
        )

        # 3. Context重建损失
        loss_context = self.mse_loss(
            pred['context'],
            pseudo_labels['context']
        )

        # 4. 物理一致性损失（如果有融合输出）
        loss_physics = 0.0
        if 'infrared_pred' in pred:
            # 融合的结果应该接近真实红外图
            loss_physics = self.mse_loss(
                pred['infrared_pred'],
                infrared_gt
            )

        # 5. 总损失
        total_loss = (
            self.lambda_intensity * loss_intensity +
            self.lambda_material * loss_material +
            self.lambda_context * loss_context +
            self.lambda_physics * loss_physics
        )

        # 返回详细损失
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_intensity': loss_intensity.item(),
            'loss_material': loss_material.item(),
            'loss_context': loss_context.item(),
            'loss_physics': loss_physics.item() if isinstance(loss_physics, torch.Tensor) else 0.0
        }

        return total_loss, loss_dict


class PhysicsConsistencyLoss(nn.Module):
    """
    物理一致性约束

    确保三个分支的组合能重建出合理的红外图
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, intensity, material_weight, infrared_pred, infrared_gt):
        """
        检查物理合理性

        约束:
        1. 高intensity区域应该在红外图中更亮
        2. 不同material应该有不同的辐射特性
        """
        # 约束1: intensity和红外亮度正相关
        infrared_brightness = infrared_gt.mean(dim=1, keepdim=True)

        # Pearson相关系数
        intensity_flat = intensity.view(intensity.size(0), -1)
        brightness_flat = infrared_brightness.view(infrared_brightness.size(0), -1)

        # 标准化
        intensity_norm = (intensity_flat - intensity_flat.mean(dim=1, keepdim=True)) / \
                        (intensity_flat.std(dim=1, keepdim=True) + 1e-8)
        brightness_norm = (brightness_flat - brightness_flat.mean(dim=1, keepdim=True)) / \
                         (brightness_flat.std(dim=1, keepdim=True) + 1e-8)

        # 相关系数（应该接近1）
        correlation = (intensity_norm * brightness_norm).mean(dim=1)
        loss_correlation = (1 - correlation).mean()

        return loss_correlation
```

### 1.5 预训练脚本

**文件**: `train/scripts/pretrain_decomposition.py`

```python
"""
预训练多任务分解网络
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from train.src.decomposition.model import MultiTaskDecompositionNet
from train.src.decomposition.pseudo_labels import prepare_pseudo_labels
from train.src.decomposition.losses import MultiTaskPretrainLoss
from train.src.train.data import YourDataset  # 你的数据集


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints/decomposition_pretrain')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0.0
    loss_dict_epoch = {
        'loss_intensity': 0.0,
        'loss_material': 0.0,
        'loss_context': 0.0,
        'loss_physics': 0.0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # 数据到设备
        visible = batch['visible'].to(device)
        infrared = batch['infrared'].to(device)
        semantic = batch.get('semantic', None)
        if semantic is not None:
            semantic = semantic.to(device)

        # 生成伪标签
        with torch.no_grad():
            pseudo_labels = prepare_pseudo_labels({
                'visible': visible,
                'infrared': infrared,
                'semantic': semantic
            })
            for key in pseudo_labels:
                pseudo_labels[key] = pseudo_labels[key].to(device)

        # 前向传播
        pred = model(visible, return_fusion=True)

        # 计算损失
        loss, loss_dict = criterion(pred, pseudo_labels, infrared)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积损失
        total_loss += loss.item()
        for key in loss_dict:
            if key != 'loss_total':
                loss_dict_epoch[key] += loss_dict[key]

        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'loss_int': loss_dict['loss_intensity'],
            'loss_mat': loss_dict['loss_material']
        })

    # 平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_dict_epoch:
        loss_dict_epoch[key] /= num_batches

    return avg_loss, loss_dict_epoch


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # 创建模型
    model = MultiTaskDecompositionNet(
        backbone=args.backbone,
        num_materials=32,
        num_context_channels=8,
        pretrained=True
    ).to(device)

    # 损失函数
    criterion = MultiTaskPretrainLoss(
        lambda_intensity=1.0,
        lambda_material=0.5,
        lambda_context=0.3,
        lambda_physics=0.5
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # 数据加载器
    # 这里需要适配你的数据集
    train_dataset = YourDataset(args.data_path, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'[INFO] Training dataset size: {len(train_dataset)}')
    print(f'[INFO] Number of batches: {len(train_loader)}')

    # 训练循环
    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f'\n[Epoch {epoch+1}/{args.epochs}]')

        # 训练
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 打印损失
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'  - Intensity: {loss_dict["loss_intensity"]:.4f}')
        print(f'  - Material: {loss_dict["loss_material"]:.4f}')
        print(f'  - Context: {loss_dict["loss_context"]:.4f}')
        print(f'  - Physics: {loss_dict["loss_physics"]:.4f}')

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'[INFO] Saved best model to {checkpoint_path}')

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

        # 学习率调度
        scheduler.step()
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

    print('\n[INFO] Training completed!')
    print(f'[INFO] Best loss: {best_loss:.4f}')
    print(f'[INFO] Model saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
```

---

## 阶段2: 集成到FLUX生成

### 2.1 集成架构

**关键点**: 预训练的分解网络输出 → FLUX的引导信号

```python
训练流程:
1. 冻结预训练的分解网络（或微调）
2. 用分解网络提取{intensity, material, context}
3. 将这些特征注入FLUX
4. 训练FLUX生成红外图
```

### 2.2 修改FLUX模型

**文件**: `train/src/train/model_decomposition.py`

```python
"""
集成分解网络的FLUX模型
"""

import lightning as L
import torch
import torch.nn as nn
from diffusers.pipelines import FluxFillPipeline

from ..decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion
from ..flux.semantic_cross_attention import SemanticCrossAttention


class DecompositionGuidedFluxModel(L.LightningModule):
    """
    集成多任务分解网络的FLUX模型
    """

    def __init__(
        self,
        flux_fill_id: str,
        decomposition_checkpoint: str = None,  # 预训练的分解网络
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        freeze_decomposition: bool = True,  # 是否冻结分解网络
    ):
        super().__init__()

        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # 加载FLUX
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)
        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2

        # 冻结FLUX的text encoder和VAE
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        # 加载预训练的分解网络
        print('[INFO] Loading pretrained decomposition network...')
        self.decomposition_net = MultiTaskDecompositionNet(
            backbone='resnet50',
            num_materials=32,
            num_context_channels=8,
            pretrained=False  # 从checkpoint加载
        )

        if decomposition_checkpoint and os.path.exists(decomposition_checkpoint):
            checkpoint = torch.load(decomposition_checkpoint, map_location=device)
            self.decomposition_net.load_state_dict(checkpoint['model_state_dict'])
            print(f'[INFO] Loaded decomposition checkpoint from: {decomposition_checkpoint}')
        else:
            print('[WARNING] No decomposition checkpoint provided, using random initialization')

        # 是否冻结分解网络
        if freeze_decomposition:
            for param in self.decomposition_net.parameters():
                param.requires_grad = False
            self.decomposition_net.eval()
            print('[INFO] Decomposition network is FROZEN')
        else:
            print('[INFO] Decomposition network will be FINE-TUNED')

        # 保持分解网络在float32（不要转到bfloat16）
        self.decomposition_net = self.decomposition_net.float()

        # 创建三个独立的cross-attention模块
        self.intensity_cross_attn = SemanticCrossAttention(dim=64, num_heads=8)
        self.material_cross_attn = SemanticCrossAttention(dim=64, num_heads=8)
        self.context_cross_attn = SemanticCrossAttention(dim=64, num_heads=8)

        # 或者使用融合的引导
        self.use_separate_guidance = model_config.get('use_separate_guidance', True)

        if not self.use_separate_guidance:
            # 单个cross-attention，输入融合特征
            self.unified_cross_attn = SemanticCrossAttention(dim=64, num_heads=8)
            # 特征融合层
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(1 + 32 + 8, 64, 1),  # intensity(1) + material(32) + context(8)
                nn.ReLU(),
                nn.Conv2d(64, 64, 1)
            )

        # LoRA
        if lora_config:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            self.lora_layers = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))

        self.to(device).to(dtype)

        # 确保分解网络保持float32
        self.decomposition_net = self.decomposition_net.float()

    def encode_decomposition_features(self, features_dict, latent_shape):
        """
        将分解网络的输出编码为FLUX可用的token序列

        Args:
            features_dict: dict with keys 'intensity', 'material_logits', 'context'
            latent_shape: FLUX潜在空间的形状 [B, C, H, W]

        Returns:
            tokens: [B, seq_len, dim] 用于cross-attention
        """
        B, C, H, W = latent_shape

        if self.use_separate_guidance:
            # 分别处理每个特征
            # Intensity: [B, 1, H_feat, W_feat] → [B, seq_len, 64]
            intensity = features_dict['intensity']
            intensity_resized = F.interpolate(intensity, size=(H, W), mode='bilinear', align_corners=True)
            intensity_tokens = intensity_resized.flatten(2).transpose(1, 2)  # [B, H*W, 1]
            intensity_tokens = intensity_tokens.repeat(1, 1, 64)  # [B, H*W, 64]

            # Material: [B, 32, H_feat, W_feat] → [B, seq_len, 64]
            material = features_dict['material_logits']
            material_resized = F.interpolate(material, size=(H, W), mode='bilinear', align_corners=True)
            material_probs = F.softmax(material_resized, dim=1)  # [B, 32, H, W]
            material_tokens = material_probs.flatten(2).transpose(1, 2)  # [B, H*W, 32]
            material_tokens = F.pad(material_tokens, (0, 32), value=0)  # [B, H*W, 64]

            # Context: [B, 8, H_feat, W_feat] → [B, seq_len, 64]
            context = features_dict['context']
            context_resized = F.interpolate(context, size=(H, W), mode='bilinear', align_corners=True)
            context_tokens = context_resized.flatten(2).transpose(1, 2)  # [B, H*W, 8]
            context_tokens = F.pad(context_tokens, (0, 56), value=0)  # [B, H*W, 64]

            return {
                'intensity_tokens': intensity_tokens,
                'material_tokens': material_tokens,
                'context_tokens': context_tokens
            }
        else:
            # 融合所有特征
            intensity = F.interpolate(features_dict['intensity'], size=(H, W), mode='bilinear')
            material = F.interpolate(features_dict['material_logits'], size=(H, W), mode='bilinear')
            context = F.interpolate(features_dict['context'], size=(H, W), mode='bilinear')

            # Concat: [B, 1+32+8, H, W]
            concat_feat = torch.cat([intensity, material, context], dim=1)

            # 融合: [B, 64, H, W]
            fused_feat = self.feature_fusion(concat_feat.float()).to(self.dtype)

            # 转为tokens: [B, H*W, 64]
            fused_tokens = fused_feat.flatten(2).transpose(1, 2)

            return {'fused_tokens': fused_tokens}

    def step(self, batch):
        """训练步骤"""
        visible = batch['visible']  # [B, 3, H, W]
        infrared = batch['infrared']

        # === 1. 通过分解网络提取特征 ===
        with torch.no_grad() if self.freeze_decomposition else torch.enable_grad():
            # 分解网络在float32
            decomp_output = self.decomposition_net(visible.float(), return_fusion=False)
            # decomp_output: {intensity, material_logits, context}

        # === 2. FLUX标准流程 ===
        with torch.no_grad():
            # 编码visible和infrared为diptych
            diptych = torch.cat([visible, infrared], dim=-1)  # [B, 3, H, 2W]

            # ... (FLUX的标准编码流程)
            # x_0, x_cond, img_ids = encode_images_fill(...)

            # 准备噪声和时间步
            # t, x_t, x_1 = ...

        # === 3. 编码分解特征为tokens ===
        latent_shape = x_0.shape  # [B, C, H_latent, W_latent]
        decomp_tokens = self.encode_decomposition_features(decomp_output, latent_shape)

        # === 4. 修改FLUX的forward，注入分解特征 ===
        # 这里需要修改transformer的forward
        # 在某些层插入cross-attention

        # 简化版本：直接加到条件上
        if self.use_separate_guidance:
            # 在FLUX的某一层后插入三个cross-attention
            # 这需要修改transformer代码
            pass
        else:
            # 单一引导
            guided_latent = self.unified_cross_attn(
                x_t_tokens,  # FLUX的潜在特征
                decomp_tokens['fused_tokens']
            )

        # === 5. FLUX生成 ===
        # pred_noise = self.transformer(...)

        # === 6. 计算损失 ===
        # loss = F.mse_loss(pred_noise, target_noise)

        return loss
```

**注意**: 这里的step()方法是简化版本，完整实现需要：
1. 参考你现有的`model.py`中的step()逻辑
2. 在合适的位置插入cross-attention

### 2.3 修改FLUX Transformer（关键）

**文件**: `train/src/flux/transformer_decomposition.py`

```python
"""
修改FLUX Transformer，支持分解特征引导
"""

# 方案1: 修改现有的transformer forward
# 在特定层后插入cross-attention

def modified_transformer_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    decomposition_tokens,  # ← 新增参数
    timestep,
    ...
):
    """
    修改FLUX的forward，插入分解特征引导
    """
    # 标准FLUX流程
    for i, block in enumerate(self.blocks):
        hidden_states = block(hidden_states, encoder_hidden_states, timestep, ...)

        # 在特定层插入cross-attention
        if i in [6, 12]:  # 中间层
            # 注入intensity引导
            hidden_states = self.intensity_cross_attn(
                hidden_states,
                decomposition_tokens['intensity_tokens']
            )

        if i == 12:  # 后期层
            # 注入material和context引导
            hidden_states = self.material_cross_attn(
                hidden_states,
                decomposition_tokens['material_tokens']
            )
            hidden_states = self.context_cross_attn(
                hidden_states,
                decomposition_tokens['context_tokens']
            )

    return hidden_states
```

### 2.4 训练配置

**文件**: `train/configs/decomposition_flux.yaml`

```yaml
# 使用分解网络引导的FLUX训练配置

model:
  flux_fill_id: "black-forest-labs/FLUX.1-Fill-dev"
  decomposition_checkpoint: "checkpoints/decomposition_pretrain/best_model.pth"
  freeze_decomposition: true  # 第一阶段冻结
  use_separate_guidance: true  # 使用三个独立的cross-attention

  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
    lora_dropout: 0.0
    bias: "none"

optimizer:
  type: "AdamW"
  params:
    lr: 1.0e-4
    weight_decay: 1.0e-4

training:
  batch_size: 4
  epochs: 100
  gradient_accumulation_steps: 2
  mixed_precision: "bf16"

  # 损失权重
  lambda_generation: 1.0
  lambda_decomposition_consistency: 0.1  # 分解网络的输出也要一致
```

### 2.5 训练脚本

**文件**: `train/scripts/train_decomposition_flux.sh`

```bash
#!/bin/bash

# 阶段2: 训练集成了分解网络的FLUX

python train/src/train/train_decomposition.py \
    --config train/configs/decomposition_flux.yaml \
    --data_path data/train.parquet \
    --output_dir checkpoints/decomposition_flux \
    --decomposition_checkpoint checkpoints/decomposition_pretrain/best_model.pth \
    --freeze_decomposition \
    --epochs 100 \
    --batch_size 4 \
    --num_workers 4 \
    --gpus 2
```

---

## 阶段3: 微调和优化

### 3.1 解冻分解网络微调

```bash
# 在阶段2训练稳定后，解冻分解网络联合微调

python train/src/train/train_decomposition.py \
    --config train/configs/decomposition_flux.yaml \
    --resume checkpoints/decomposition_flux/checkpoint_epoch_50.pth \
    --no_freeze_decomposition \  # 解冻
    --lr 1e-5 \  # 降低学习率
    --epochs 150
```

### 3.2 可视化分析

**文件**: `train/scripts/visualize_decomposition.py`

```python
"""
可视化分解网络的输出
"""

import torch
import matplotlib.pyplot as plt
from train.src.decomposition.model import MultiTaskDecompositionNet

def visualize_decomposition(model, image_path):
    """
    可视化分解结果
    """
    # 加载图像
    rgb = load_image(image_path)  # [1, 3, H, W]

    # 前向传播
    with torch.no_grad():
        output = model(rgb.cuda(), return_fusion=True)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原图
    axes[0, 0].imshow(rgb[0].permute(1, 2, 0).cpu())
    axes[0, 0].set_title('RGB Input')

    # Intensity
    axes[0, 1].imshow(output['intensity'][0, 0].cpu(), cmap='hot')
    axes[0, 1].set_title('Intensity (类温度)')

    # Material
    material_vis = torch.argmax(output['material_logits'][0], dim=0).cpu()
    axes[0, 2].imshow(material_vis, cmap='tab20')
    axes[0, 2].set_title('Material (材质)')

    # Context (取前3个通道可视化)
    context_vis = output['context'][0, :3].permute(1, 2, 0).cpu()
    axes[1, 0].imshow(context_vis)
    axes[1, 0].set_title('Context Features')

    # 融合的红外预测
    axes[1, 1].imshow(output['infrared_pred'][0].permute(1, 2, 0).cpu())
    axes[1, 1].set_title('Predicted Infrared (from fusion)')

    plt.tight_layout()
    plt.savefig('decomposition_visualization.png')
    print('Saved to decomposition_visualization.png')
```

---

## 总结：完整训练流程

```bash
# === 阶段1: 预训练分解网络 (2-3周) ===
python train/scripts/pretrain_decomposition.py \
    --data_path data/train.parquet \
    --output_dir checkpoints/decomposition_pretrain \
    --epochs 50 \
    --batch_size 8

# 可视化分解结果
python train/scripts/visualize_decomposition.py \
    --checkpoint checkpoints/decomposition_pretrain/best_model.pth \
    --image_path data/test/sample.jpg

# === 阶段2: 集成到FLUX训练 (2-3周) ===
# 冻结分解网络
python train/scripts/train_decomposition_flux.sh

# === 阶段3: 联合微调 (1-2周) ===
# 解冻分解网络
python train/src/train/train_decomposition.py \
    --resume checkpoints/decomposition_flux/checkpoint_epoch_50.pth \
    --no_freeze_decomposition \
    --lr 1e-5 \
    --epochs 150

# === 阶段4: 评估和可视化 ===
python train/scripts/evaluate_decomposition.py \
    --checkpoint checkpoints/decomposition_flux_finetuned/best_model.pth \
    --test_data data/test.parquet
```

---

## 下一步

我已经给你提供了完整的实现框架，包括:

1. ✅ 预训练阶段的完整代码
2. ✅ 伪标签生成逻辑
3. ✅ 多任务损失函数
4. ✅ FLUX集成架构
5. ✅ 训练脚本和配置

**你现在需要**:
1. 告诉我你想先从哪个阶段开始？
2. 我可以帮你完善具体的代码细节
3. 或者我可以先帮你实现一个简化版本快速验证

你觉得如何？需要我详细实现哪一部分？
