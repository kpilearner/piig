"""
诊断脚本：检查训练流程中的数据和模型

用于排查loss不降的问题
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.parquet_dataloader import create_dataloaders
from decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion
from decomposition.pseudo_labels import (
    generate_pseudo_intensity,
    generate_pseudo_material_from_panoptic,
    generate_pseudo_context
)
import matplotlib.pyplot as plt
import numpy as np

def check_data_loading():
    """检查数据加载"""
    print("=" * 70)
    print("1. 检查数据加载")
    print("=" * 70)

    # 创建dataloader
    train_loader, _ = create_dataloaders(
        parquet_path="/root/autodl-tmp/qyt_contrasctivew/lmr_contrastive/dataset/pid_llvip_dataset_fixed.parquet",
        split_file="./data/dataset_split.json",
        batch_size=2,
        num_workers=0,
        image_size=512,
        normalize=True,
    )

    # 加载一个batch
    batch = next(iter(train_loader))

    print(f"\n✅ 数据形状:")
    print(f"   RGB: {batch['rgb'].shape}")
    print(f"   Infrared: {batch['infrared'].shape}")
    print(f"   Semantic: {batch['semantic'].shape}")

    print(f"\n✅ 数据范围:")
    print(f"   RGB: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
    print(f"   Infrared: [{batch['infrared'].min():.3f}, {batch['infrared'].max():.3f}]")
    print(f"   Semantic: [{batch['semantic'].min():.3f}, {batch['semantic'].max():.3f}]")

    # 检查是否有NaN
    print(f"\n✅ NaN检查:")
    print(f"   RGB有NaN: {torch.isnan(batch['rgb']).any()}")
    print(f"   Infrared有NaN: {torch.isnan(batch['infrared']).any()}")
    print(f"   Semantic有NaN: {torch.isnan(batch['semantic']).any()}")

    return batch


def check_pseudo_labels(batch):
    """检查伪标签生成"""
    print("\n" + "=" * 70)
    print("2. 检查伪标签生成")
    print("=" * 70)

    rgb = batch['rgb']
    infrared = batch['infrared']
    semantic = batch['semantic']

    # 生成伪标签
    ir_gray = infrared.mean(dim=1, keepdim=True)

    print(f"\n✅ 红外灰度图:")
    print(f"   形状: {ir_gray.shape}")
    print(f"   范围: [{ir_gray.min():.3f}, {ir_gray.max():.3f}]")

    # Intensity
    pseudo_intensity = generate_pseudo_intensity(ir_gray, normalize=True)
    print(f"\n✅ Pseudo Intensity:")
    print(f"   形状: {pseudo_intensity.shape}")
    print(f"   范围: [{pseudo_intensity.min():.3f}, {pseudo_intensity.max():.3f}]")
    print(f"   均值: {pseudo_intensity.mean():.3f}")
    print(f"   标准差: {pseudo_intensity.std():.3f}")

    # Material
    pseudo_material = generate_pseudo_material_from_panoptic(semantic, num_classes=32)
    print(f"\n✅ Pseudo Material:")
    print(f"   形状: {pseudo_material.shape}")
    print(f"   范围: [{pseudo_material.min()}, {pseudo_material.max()}]")
    print(f"   唯一类别数: {len(torch.unique(pseudo_material))}")

    # Context
    pseudo_context = generate_pseudo_context(rgb, ir_gray)
    print(f"\n✅ Pseudo Context:")
    print(f"   形状: {pseudo_context.shape}")
    print(f"   范围: [{pseudo_context.min():.3f}, {pseudo_context.max():.3f}]")

    return {
        'pseudo_intensity': pseudo_intensity,
        'pseudo_material': pseudo_material,
        'pseudo_context': pseudo_context,
    }


def check_model_forward(batch, device='cuda'):
    """检查模型前向传播"""
    print("\n" + "=" * 70)
    print("3. 检查模型前向传播")
    print("=" * 70)

    # 创建模型
    model = MultiTaskDecompositionNet(
        backbone='resnet50',
        pretrained=True,
        num_material_classes=32,
        context_channels=8,
    ).to(device)

    fusion = PhysicsInspiredFusion(
        num_material_classes=32,
        context_channels=8,
        hidden_dim=64,
    ).to(device)

    model.eval()
    fusion.eval()

    rgb = batch['rgb'].to(device)

    # 前向传播
    with torch.no_grad():
        pred_intensity, pred_material, pred_context = model(rgb)
        fused_output = fusion(pred_intensity, pred_material, pred_context)

    print(f"\n✅ 模型输出:")
    print(f"   Intensity: {pred_intensity.shape}, 范围: [{pred_intensity.min():.3f}, {pred_intensity.max():.3f}]")
    print(f"   Material: {pred_material.shape}, 范围: [{pred_material.min():.3f}, {pred_material.max():.3f}]")
    print(f"   Context: {pred_context.shape}, 范围: [{pred_context.min():.3f}, {pred_context.max():.3f}]")
    print(f"   Fused: {fused_output.shape}, 范围: [{fused_output.min():.3f}, {fused_output.max():.3f}]")

    # 检查梯度
    model.train()
    fusion.train()

    pred_intensity, pred_material, pred_context = model(rgb)
    fused_output = fusion(pred_intensity, pred_material, pred_context)

    # 简单loss
    loss = fused_output.mean()
    loss.backward()

    # 检查是否有梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"\n✅ 梯度检查 (示例): {name}")
            print(f"   梯度范数: {param.grad.norm():.6f}")
            break

    if not has_grad:
        print("\n❌ 警告: 模型没有梯度！")

    return {
        'pred_intensity': pred_intensity.detach().cpu(),
        'pred_material': pred_material.detach().cpu(),
        'pred_context': pred_context.detach().cpu(),
        'fused_output': fused_output.detach().cpu(),
    }


def check_loss_computation(batch, pseudo_labels, predictions):
    """检查损失计算"""
    print("\n" + "=" * 70)
    print("4. 检查损失计算")
    print("=" * 70)

    from decomposition.losses import MultiTaskPretrainLoss

    criterion = MultiTaskPretrainLoss(
        lambda_intensity=1.0,
        lambda_material=1.0,
        lambda_context=0.5,
        lambda_fusion=2.0,
    )

    # 准备数据
    ir_gray = batch['infrared'].mean(dim=1, keepdim=True)

    # 计算损失
    loss, losses = criterion(
        predictions['pred_intensity'],
        predictions['pred_material'],
        predictions['pred_context'],
        predictions['fused_output'],
        pseudo_labels['pseudo_intensity'],
        pseudo_labels['pseudo_material'],
        pseudo_labels['pseudo_context'],
        ir_gray
    )

    print(f"\n✅ 各项损失:")
    print(f"   Total: {losses['total']:.4f}")
    print(f"   Intensity: {losses['intensity']:.4f}")
    print(f"   Material: {losses['material']:.4f}")
    print(f"   Context: {losses['context']:.4f}")
    print(f"   Fusion: {losses['fusion']:.4f}")

    # 分析损失大小
    if losses['total'] > 100:
        print(f"\n⚠️  警告: Total loss很大 ({losses['total']:.2f})，可能有数值问题")

    if losses['material'] > 10:
        print(f"\n⚠️  警告: Material loss很大 ({losses['material']:.2f})，检查类别数量")

    return losses


def visualize_outputs(batch, pseudo_labels, predictions, save_path='debug_vis.png'):
    """可视化对比"""
    print("\n" + "=" * 70)
    print("5. 可视化输出")
    print("=" * 70)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # 第一行：输入
    axes[0, 0].imshow(batch['rgb'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axes[0, 0].set_title('Input RGB')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(batch['infrared'][0].permute(1, 2, 0).numpy())
    axes[0, 1].set_title('Target Infrared')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(batch['semantic'][0].permute(1, 2, 0).numpy())
    axes[0, 2].set_title('Semantic')
    axes[0, 2].axis('off')

    axes[0, 3].axis('off')

    # 第二行：伪标签
    axes[1, 0].imshow(pseudo_labels['pseudo_intensity'][0, 0].numpy(), cmap='hot')
    axes[1, 0].set_title('Pseudo Intensity')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pseudo_labels['pseudo_material'][0].numpy(), cmap='tab20')
    axes[1, 1].set_title('Pseudo Material')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pseudo_labels['pseudo_context'][0, 0].numpy(), cmap='viridis')
    axes[1, 2].set_title('Pseudo Context (ch0)')
    axes[1, 2].axis('off')

    axes[1, 3].axis('off')

    # 第三行：预测
    axes[2, 0].imshow(predictions['pred_intensity'][0, 0].numpy(), cmap='hot')
    axes[2, 0].set_title('Pred Intensity')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(predictions['pred_material'][0].argmax(0).numpy(), cmap='tab20')
    axes[2, 1].set_title('Pred Material')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(predictions['pred_context'][0, 0].numpy(), cmap='viridis')
    axes[2, 2].set_title('Pred Context (ch0)')
    axes[2, 2].axis('off')

    axes[2, 3].imshow(predictions['fused_output'][0, 0].numpy(), cmap='hot')
    axes[2, 3].set_title('Fused Output')
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化保存到: {save_path}")


def main():
    print("\n" + "=" * 70)
    print("🔍 训练诊断脚本")
    print("=" * 70)

    # 1. 检查数据
    batch = check_data_loading()

    # 2. 检查伪标签
    pseudo_labels = check_pseudo_labels(batch)

    # 3. 检查模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions = check_model_forward(batch, device)

    # 4. 检查损失
    losses = check_loss_computation(batch, pseudo_labels, predictions)

    # 5. 可视化
    visualize_outputs(batch, pseudo_labels, predictions)

    print("\n" + "=" * 70)
    print("✅ 诊断完成！")
    print("=" * 70)
    print(f"\n如果loss不降，检查：")
    print(f"1. 数据范围是否正确 (RGB归一化, Infrared [0,1])")
    print(f"2. 伪标签是否合理")
    print(f"3. 模型输出范围是否匹配伪标签")
    print(f"4. 损失数值是否在合理范围 (total < 10)")
    print(f"5. 查看 debug_vis.png 检查可视化")


if __name__ == '__main__':
    main()
