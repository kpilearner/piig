"""
固定数据集划分工具

确保阶段1预训练和阶段2 FLUX训练使用完全相同的训练/验证集划分，
避免数据泄漏。

与ICEdit_contrastive保持一致的缓存策略：
- 使用环境变量 HF_DATASETS_CACHE 控制datasets缓存位置
- 使用环境变量 HUGGINGFACE_HUB_CACHE 控制模型缓存位置
"""

import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset

# 读取环境变量中的缓存路径（与ICEdit保持一致）
# 如果没有设置，使用默认位置
DATASETS_CACHE = os.environ.get('HF_DATASETS_CACHE', None)


def create_fixed_split(
    parquet_path,
    split_ratio=0.9,
    seed=42,
    output_dir='./data',
    force_recreate=False
):
    """
    创建并保存固定的数据集划分索引

    Args:
        parquet_path: parquet文件路径
        split_ratio: 训练集比例（默认0.9 = 90%训练，10%验证）
        seed: 随机种子（固定为42确保可复现）
        output_dir: 输出目录
        force_recreate: 是否强制重新创建（默认False，如果已存在则跳过）

    Returns:
        split_info: dict包含划分信息
    """
    os.makedirs(output_dir, exist_ok=True)

    # 划分信息文件路径
    split_file = os.path.join(output_dir, 'dataset_split.json')

    # 如果已存在且不强制重新创建，直接加载
    if os.path.exists(split_file) and not force_recreate:
        print(f"✅ 使用已存在的划分文件: {split_file}")
        with open(split_file, 'r') as f:
            split_info = json.load(f)

        # 验证
        print(f"   训练集大小: {len(split_info['train_indices'])}")
        print(f"   验证集大小: {len(split_info['val_indices'])}")
        print(f"   总计: {split_info['total_size']}")
        print(f"   划分比例: {split_info['split_ratio']}")
        print(f"   随机种子: {split_info['seed']}")

        return split_info

    # 使用PyArrow直接读取元数据，避免加载所有数据到内存
    print(f"📂 读取数据集元数据: {parquet_path}")
    if DATASETS_CACHE:
        print(f"   缓存路径: {DATASETS_CACHE}")

    # 只读取元数据获取行数，不加载任何实际数据
    parquet_file = pq.ParquetFile(parquet_path)
    total_size = parquet_file.metadata.num_rows
    print(f"   总样本数: {total_size}")

    # 计算划分点
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    print(f"🔀 划分数据集 (ratio={split_ratio}, seed={seed})...")
    print(f"   训练集: {train_size} 样本")
    print(f"   验证集: {val_size} 样本")

    # 生成索引（使用固定种子）
    np.random.seed(seed)
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)

    train_indices = sorted(all_indices[:train_size].tolist())
    val_indices = sorted(all_indices[train_size:].tolist())

    # 保存划分信息
    split_info = {
        'parquet_path': str(Path(parquet_path).absolute()),
        'total_size': total_size,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'split_ratio': split_ratio,
        'seed': seed,
        'created_at': str(Path(split_file).stat().st_mtime if os.path.exists(split_file) else 'new')
    }

    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"✅ 划分信息已保存到: {split_file}")
    print(f"   训练集: {split_info['train_size']} 样本")
    print(f"   验证集: {split_info['val_size']} 样本")

    return split_info


def load_split_dataset(
    parquet_path,
    split_file='./data/dataset_split.json',
    create_if_not_exists=True
):
    """
    加载使用固定划分的数据集

    Args:
        parquet_path: parquet文件路径
        split_file: 划分信息文件路径
        create_if_not_exists: 如果划分文件不存在，是否自动创建

    Returns:
        train_dataset, val_dataset: 训练和验证数据集
    """
    # 检查划分文件是否存在
    if not os.path.exists(split_file):
        if create_if_not_exists:
            print(f"⚠️  划分文件不存在，自动创建...")
            split_info = create_fixed_split(
                parquet_path,
                output_dir=os.path.dirname(split_file) or './data'
            )
        else:
            raise FileNotFoundError(
                f"划分文件不存在: {split_file}\n"
                f"请先运行 create_fixed_split() 创建划分"
            )
    else:
        # 加载划分信息
        with open(split_file, 'r') as f:
            split_info = json.load(f)

        print(f"✅ 加载划分信息: {split_file}")

    # 加载完整数据集（使用环境变量中的缓存路径）
    print(f"📂 加载数据集: {parquet_path}")
    dataset = load_dataset(
        'parquet',
        data_files=parquet_path,
        cache_dir=DATASETS_CACHE  # 使用与ICEdit相同的缓存策略
    )
    full_dataset = dataset['train']

    # 验证数据集大小
    if len(full_dataset) != split_info['total_size']:
        print(f"⚠️  警告: 数据集大小不匹配!")
        print(f"   期望: {split_info['total_size']}")
        print(f"   实际: {len(full_dataset)}")
        print(f"   建议重新创建划分文件")

    # 使用保存的索引进行划分
    train_indices = split_info['train_indices']
    val_indices = split_info['val_indices']

    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)

    print(f"📊 数据集加载完成:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")

    return train_dataset, val_dataset


def verify_split_consistency(split_file='./data/dataset_split.json'):
    """
    验证数据集划分的一致性

    检查JSON文件的完整性和索引的合理性
    """
    print("🔍 验证数据集划分一致性...")

    if not os.path.exists(split_file):
        print(f"❌ 划分文件不存在: {split_file}")
        return False

    with open(split_file, 'r') as f:
        split_info = json.load(f)

    # 验证必需字段
    required_fields = ['parquet_path', 'total_size', 'train_size', 'val_size',
                       'train_indices', 'val_indices', 'split_ratio', 'seed']
    for field in required_fields:
        if field not in split_info:
            print(f"❌ 缺少必需字段: {field}")
            return False

    # 验证数据一致性
    train_indices = split_info['train_indices']
    val_indices = split_info['val_indices']

    if len(train_indices) != split_info['train_size']:
        print(f"❌ 训练集索引数量不匹配: {len(train_indices)} vs {split_info['train_size']}")
        return False

    if len(val_indices) != split_info['val_size']:
        print(f"❌ 验证集索引数量不匹配: {len(val_indices)} vs {split_info['val_size']}")
        return False

    if len(train_indices) + len(val_indices) != split_info['total_size']:
        print(f"❌ 总样本数不匹配")
        return False

    # 验证索引无重叠
    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        print(f"❌ 训练集和验证集有重叠")
        return False

    # 验证索引范围
    all_indices = train_set | val_set
    if min(all_indices) < 0 or max(all_indices) >= split_info['total_size']:
        print(f"❌ 索引超出范围")
        return False

    print("✅ 数据集划分一致性验证通过!")
    print(f"   训练集: {split_info['train_size']} 样本")
    print(f"   验证集: {split_info['val_size']} 样本")
    print(f"   总计: {split_info['total_size']} 样本")
    print(f"   种子: {split_info['seed']}")
    print(f"   比例: {split_info['split_ratio']}")

    return True


# ============================================================================
# 使用示例
# ============================================================================
"""
与ICEdit_contrastive保持一致的缓存设置：

方法1: 在命令行设置环境变量（推荐）
    export HF_DATASETS_CACHE="/root/autodl-tmp/.cache"
    export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/.cache"
    python utils/fixed_split.py --parquet /path/to/data.parquet

方法2: 在训练脚本中设置（与ICEdit train_joint.sh相同）
    #!/bin/bash
    DATA_CACHE="/root/autodl-tmp/.cache"
    export HF_DATASETS_CACHE="$DATA_CACHE"
    export HUGGINGFACE_HUB_CACHE="$DATA_CACHE"
    python scripts/pretrain_decomposition.py

方法3: 在Python代码中设置
    import os
    os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.cache'
    from utils.fixed_split import load_split_dataset
    train_ds, val_ds = load_split_dataset('/path/to/data.parquet')

这样确保：
1. 数据集缓存不会重复下载
2. 阶段1和阶段2使用相同的缓存位置
3. 节省磁盘空间
"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='创建固定的数据集划分')
    parser.add_argument('--parquet', type=str, required=True,
                       help='Parquet文件路径')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='输出目录')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                       help='训练集比例 (默认0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认42)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新创建划分')
    parser.add_argument('--verify', action='store_true',
                       help='验证划分一致性')

    args = parser.parse_args()

    # 创建固定划分
    split_info = create_fixed_split(
        parquet_path=args.parquet,
        split_ratio=args.split_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        force_recreate=args.force
    )

    print("\n" + "="*60)
    print("📋 划分摘要:")
    print("="*60)
    print(f"Parquet路径: {split_info['parquet_path']}")
    print(f"总样本数: {split_info['total_size']}")
    print(f"训练集: {split_info['train_size']} ({split_info['train_size']/split_info['total_size']*100:.1f}%)")
    print(f"验证集: {split_info['val_size']} ({split_info['val_size']/split_info['total_size']*100:.1f}%)")
    print(f"随机种子: {split_info['seed']}")
    print(f"划分文件: {args.output_dir}/dataset_split.json")
    print("="*60)

    # 验证（如果请求）
    if args.verify:
        print("\n")
        verify_split_consistency(f"{args.output_dir}/dataset_split.json")

    print("\n✅ 完成!")
    print("\n💡 使用方法:")
    print(f"   在训练脚本中导入: from utils.fixed_split import load_split_dataset")
    print(f"   加载数据集: train_ds, val_ds = load_split_dataset('{args.parquet}')")
