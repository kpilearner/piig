"""
Parquet数据集加载器 - 用于阶段1分解网络预训练

与ICEdit_contrastive的数据格式完全兼容：
- 加载parquet文件
- 提取 src_img (可见光), edited_img (红外), panoptic_img (语义)
- 使用fixed_split.py生成的划分
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os

# 导入fixed_split工具
from .fixed_split import load_split_dataset


class DecompositionDataset(Dataset):
    """
    用于分解网络预训练的数据集

    从parquet加载三模态数据：可见光、红外、语义分割
    """
    def __init__(
        self,
        dataset,  # 从load_split_dataset()返回的HF dataset对象
        image_size=512,
        normalize=True,
    ):
        """
        Args:
            dataset: HuggingFace Dataset对象（已经过select划分）
            image_size: 图像resize尺寸
            normalize: 是否使用ImageNet归一化（用于ResNet backbone）
        """
        self.dataset = dataset
        self.image_size = image_size

        # 图像变换
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            # ImageNet归一化（ResNet预训练使用）
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self.transform = transforms.Compose(transform_list)

        # 语义图不需要归一化，保持原始颜色
        self.semantic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        返回:
            rgb: [3, H, W] 可见光图像（ImageNet归一化）
            infrared: [3, H, W] 红外图像（仅ToTensor，不归一化！）
            semantic: [3, H, W] 语义分割图（仅ToTensor，保留颜色信息）
        """
        sample = self.dataset[idx]

        # 加载图像
        rgb_img = sample['src_img']  # PIL Image
        ir_img = sample['edited_img']  # PIL Image
        semantic_img = sample['panoptic_img']  # PIL Image (颜色分块)

        # 确保是RGB模式
        if rgb_img.mode != 'RGB':
            rgb_img = rgb_img.convert('RGB')
        if ir_img.mode != 'RGB':
            ir_img = ir_img.convert('RGB')
        if semantic_img.mode != 'RGB':
            semantic_img = semantic_img.convert('RGB')

        # 应用变换
        rgb = self.transform(rgb_img)  # ImageNet归一化（ResNet需要）

        # ⚠️ 红外图不归一化！保持 [0, 1] 范围
        infrared = self.semantic_transform(ir_img)  # 只做 Resize + ToTensor

        semantic = self.semantic_transform(semantic_img)  # 只做 Resize + ToTensor

        return {
            'rgb': rgb,
            'infrared': infrared,
            'semantic': semantic,
        }


def create_dataloaders(
    parquet_path,
    split_file='./data/dataset_split.json',
    batch_size=8,
    num_workers=4,
    image_size=512,
    normalize=True,
):
    """
    创建训练和验证数据加载器

    Args:
        parquet_path: parquet文件路径
        split_file: 数据集划分JSON文件路径
        batch_size: 批次大小
        num_workers: DataLoader工作进程数
        image_size: 图像尺寸
        normalize: 是否归一化

    Returns:
        train_loader, val_loader
    """
    # 使用环境变量中的缓存路径（与ICEdit一致）
    cache_dir = os.environ.get('HF_DATASETS_CACHE', None)
    if cache_dir:
        print(f"📦 使用缓存路径: {cache_dir}")

    # 加载已划分的数据集
    print(f"📂 加载数据集: {parquet_path}")
    train_dataset_hf, val_dataset_hf = load_split_dataset(
        parquet_path=parquet_path,
        split_file=split_file,
        create_if_not_exists=True
    )

    # 创建PyTorch Dataset
    print("🔧 创建PyTorch数据集...")
    train_dataset = DecompositionDataset(
        train_dataset_hf,
        image_size=image_size,
        normalize=normalize,
    )

    val_dataset = DecompositionDataset(
        val_dataset_hf,
        image_size=image_size,
        normalize=normalize,
    )

    # 创建DataLoader
    print("🔧 创建DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"✅ 数据加载器创建完成:")
    print(f"   训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"   验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")

    return train_loader, val_loader


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试数据加载器')
    parser.add_argument('--parquet', type=str, required=True,
                       help='Parquet文件路径')
    parser.add_argument('--split_file', type=str, default='./data/dataset_split.json',
                       help='数据集划分文件路径')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')

    args = parser.parse_args()

    print("=" * 70)
    print("测试Parquet数据加载器")
    print("=" * 70)

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        parquet_path=args.parquet,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=0,  # 测试时使用单进程
        image_size=512,
    )

    # 加载一个批次测试
    print("\n" + "=" * 70)
    print("加载测试批次...")
    print("=" * 70)

    batch = next(iter(train_loader))

    print(f"\n✅ 批次加载成功!")
    print(f"   RGB形状: {batch['rgb'].shape}")
    print(f"   红外形状: {batch['infrared'].shape}")
    print(f"   语义形状: {batch['semantic'].shape}")
    print(f"   RGB范围: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
    print(f"   红外范围: [{batch['infrared'].min():.3f}, {batch['infrared'].max():.3f}]")
    print(f"   语义范围: [{batch['semantic'].min():.3f}, {batch['semantic'].max():.3f}]")

    print("\n" + "=" * 70)
    print("✅ 数据加载器测试通过!")
    print("=" * 70)
