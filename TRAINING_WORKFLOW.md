# 🚀 完整训练流程指南

这是一个详细的、step-by-step的训练流程，从环境准备到最终训练完成。

---

## 📋 总览

```
阶段0: 环境准备 (1-2小时)
   ↓
阶段1: 预训练分解网络 (6-8小时，独立训练)
   ↓
阶段2: 集成到FLUX训练 (2-3天，主训练)
   ↓
阶段3: 端到端微调 (12小时，可选)
```

---

## 🔧 阶段0: 环境准备

### **步骤0.1: 创建Python环境**

```bash
# 创建conda环境
conda create -n physics_ir python=3.10 -y
conda activate physics_ir

# 验证Python版本
python --version  # 应该显示 Python 3.10.x
```

---

### **步骤0.2: 安装PyTorch**

```bash
# 根据您的CUDA版本选择（这里以CUDA 12.1为例）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 预期输出:
# PyTorch: 2.x.x
# CUDA available: True
# CUDA version: 12.1
```

---

### **步骤0.3: 安装核心依赖**

```bash
# Diffusion相关
pip install diffusers==0.25.0
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install peft==0.7.0

# 训练框架
pip install lightning==2.1.0
pip install prodigyopt==1.0

# 数据处理
pip install datasets==2.16.0
pip install pyarrow==14.0.0
pip install pillow==10.0.0
pip install opencv-python==4.8.0
pip install scikit-learn==1.3.0

# 日志和可视化
pip install tensorboard==2.15.0
pip install wandb==0.16.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
pip install tqdm

# 验证安装
pip list | grep -E "diffusers|transformers|lightning|datasets"
```

---

### **步骤0.4: HuggingFace认证**

```bash
# 登录HuggingFace（FLUX模型需要）
huggingface-cli login

# 输入您的token（从 https://huggingface.co/settings/tokens 获取）
# 选择 'y' 保存token

# 验证登录
huggingface-cli whoami
```

---

### **步骤0.5: 数据检查**

```bash
# 检查您的parquet数据集
python -c "
from datasets import load_dataset
import sys

# 修改为您的实际路径
parquet_path = '/root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet'

try:
    dataset = load_dataset('parquet', data_files=parquet_path)
    print('✅ 数据集加载成功')
    print(f'Columns: {dataset[\"train\"].column_names}')
    print(f'Total samples: {len(dataset[\"train\"])}')

    # 检查第一个样本
    sample = dataset['train'][0]
    print(f'\\n样本检查:')
    print(f'  src_img: {type(sample[\"src_img\"])}, size: {sample[\"src_img\"].size}')
    print(f'  edited_img: {type(sample[\"edited_img\"])}, size: {sample[\"edited_img\"].size}')
    if 'panoptic_img' in sample:
        print(f'  panoptic_img: {type(sample[\"panoptic_img\"])}, size: {sample[\"panoptic_img\"].size}')
    else:
        print('  ⚠️ 警告: 没有panoptic_img列')

except Exception as e:
    print(f'❌ 错误: {e}')
    sys.exit(1)
"
```

**预期输出:**
```
✅ 数据集加载成功
Columns: ['src_img', 'edited_img', 'panoptic_img', 'edited_prompt_list', 'task']
Total samples: 10000

样本检查:
  src_img: <class 'PIL.Image.Image'>, size: (640, 480)
  edited_img: <class 'PIL.Image.Image'>, size: (640, 480)
  panoptic_img: <class 'PIL.Image.Image'>, size: (640, 480)
```

---

### **步骤0.6: 创建数据适配器**

创建文件 `physics_inspired_infrared_generation/utils/parquet_adapter.py`:

```bash
cd physics_inspired_infrared_generation
touch utils/parquet_adapter.py
```

然后将以下代码复制到该文件：

```python
"""
Parquet数据集适配器
将您的parquet数据适配到分解网络训练格式
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class ParquetToDecompositionDataset(Dataset):
    """
    适配器: parquet → 分解网络训练格式
    """
    def __init__(self, parquet_dataset, condition_size=512, use_panoptic=True):
        self.dataset = parquet_dataset
        self.condition_size = condition_size
        self.use_panoptic = use_panoptic

        # RGB transform (ImageNet normalization for ResNet)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((condition_size, condition_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # IR and semantic transform (no normalization)
        self.basic_transform = transforms.Compose([
            transforms.Resize((condition_size, condition_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Load images
        visible_pil = sample['src_img'].convert('RGB')
        infrared_pil = sample['edited_img'].convert('RGB')  # Keep 3 channels

        # Transform
        visible = self.rgb_transform(visible_pil)
        infrared = self.basic_transform(infrared_pil)

        result = {
            'rgb': visible,      # [3, 512, 512] normalized
            'infrared': infrared,  # [3, 512, 512] [0,1]
            'image_id': str(idx)
        }

        # Add panoptic segmentation
        if self.use_panoptic and 'panoptic_img' in sample:
            panoptic_pil = sample['panoptic_img'].convert('RGB')
            panoptic = self.basic_transform(panoptic_pil)
            result['semantic'] = panoptic  # [3, 512, 512]

        return result


def create_dataloaders_from_parquet(
    parquet_path,
    batch_size=8,
    num_workers=4,
    train_split=0.9,
    use_panoptic=True
):
    """
    从parquet文件创建训练和验证数据加载器
    """
    from datasets import load_dataset

    # Load parquet
    dataset = load_dataset('parquet', data_files=parquet_path)
    full_dataset = dataset['train']

    # Split train/val
    split_dataset = full_dataset.train_test_split(
        train_size=train_split,
        seed=42
    )

    # Create datasets
    train_dataset = ParquetToDecompositionDataset(
        split_dataset['train'],
        use_panoptic=use_panoptic
    )
    val_dataset = ParquetToDecompositionDataset(
        split_dataset['test'],
        use_panoptic=use_panoptic
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# 测试代码
if __name__ == '__main__':
    import sys

    # 修改为您的实际路径
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else './data.parquet'

    print(f"Testing data loading from: {parquet_path}")

    train_loader, val_loader = create_dataloaders_from_parquet(
        parquet_path,
        batch_size=2,
        num_workers=0,
        use_panoptic=True
    )

    print(f"\n✅ Dataloaders created successfully")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\n📦 Batch contents:")
    print(f"  rgb: {batch['rgb'].shape}, dtype: {batch['rgb'].dtype}")
    print(f"  infrared: {batch['infrared'].shape}, dtype: {batch['infrared'].dtype}")
    if 'semantic' in batch:
        print(f"  semantic: {batch['semantic'].shape}, dtype: {batch['semantic'].dtype}")
    print(f"  image_id: {batch['image_id'][:2]}")

    print("\n✅ Data loading test passed!")
```

**测试适配器:**

```bash
# 测试数据加载（修改为您的实际路径）
python utils/parquet_adapter.py /root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet
```

---

## 🎯 阶段1: 预训练分解网络

### **目标**
独立训练分解网络，学习从RGB提取 intensity, material, context 特征。

### **步骤1.1: 修改伪标签生成器**

编辑 `decomposition/pseudo_labels.py`，在文件末尾添加：

```python
def generate_pseudo_material_from_panoptic(semantic_img, num_classes=32):
    """
    从全景分割图（颜色分块图）生成材料伪标签

    Args:
        semantic_img: [B, 3, H, W] RGB颜色分块图
        num_classes: 材料类别数

    Returns:
        pseudo_material: [B, H, W] 类别索引 (0到num_classes-1)
    """
    import numpy as np

    B, _, H, W = semantic_img.shape
    device = semantic_img.device

    pseudo_material = torch.zeros(B, H, W, dtype=torch.long, device=device)

    for i in range(B):
        # 转换为numpy [H, W, 3]
        rgb = semantic_img[i].permute(1, 2, 0).cpu().numpy()

        # 将RGB转换为颜色ID
        rgb_uint = (rgb * 255).astype(np.uint32)
        color_ids = (rgb_uint[:, :, 0] << 16) | (rgb_uint[:, :, 1] << 8) | rgb_uint[:, :, 2]

        # 获取唯一颜色
        unique_colors, inverse = np.unique(color_ids, return_inverse=True)

        # 映射到num_classes
        class_ids = (np.arange(len(unique_colors)) % num_classes)
        labels = class_ids[inverse].reshape(H, W)

        pseudo_material[i] = torch.from_numpy(labels).long().to(device)

    return pseudo_material


# 修改PseudoLabelGenerator类，在__init__中添加参数
class PseudoLabelGenerator:
    def __init__(
        self,
        num_material_classes=32,
        context_channels=8,
        material_method='kmeans',
        use_panoptic=True,  # 新增
        cache_materials=True
    ):
        self.num_material_classes = num_material_classes
        self.context_channels = context_channels
        self.material_method = material_method
        self.use_panoptic = use_panoptic  # 新增
        self.cache_materials = cache_materials
        self.material_cache = {}

    def __call__(self, rgb_image, infrared_image, semantic_image=None, image_ids=None):
        # Intensity
        pseudo_intensity = generate_pseudo_intensity(infrared_image, normalize=True)

        # Material (优先使用panoptic)
        if self.use_panoptic and semantic_image is not None:
            pseudo_material = generate_pseudo_material_from_panoptic(
                semantic_image, self.num_material_classes
            )
        else:
            # 回退到聚类
            pseudo_material = generate_pseudo_material(
                rgb_image, infrared_image,
                num_classes=self.num_material_classes,
                method=self.material_method
            )

        # Context
        pseudo_context = generate_pseudo_context(
            rgb_image, infrared_image, self.context_channels
        )

        return pseudo_intensity, pseudo_material, pseudo_context
```

---

### **步骤1.2: 修改预训练脚本**

编辑 `scripts/pretrain_decomposition.py`，在开头添加导入：

```python
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.parquet_adapter import create_dataloaders_from_parquet
```

然后在 `main()` 函数中修改数据加载部分：

```python
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # ===== 修改：使用parquet数据加载器 =====
    print("Loading datasets from parquet...")
    train_loader, val_loader = create_dataloaders_from_parquet(
        parquet_path=args.train_data,  # parquet文件路径
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.9,
        use_panoptic=True  # 使用全景分割图
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 创建模型（不变）
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

    # 创建损失函数
    loss_fn = MultiTaskPretrainLoss(
        lambda_intensity=args.lambda_intensity,
        lambda_material=args.lambda_material,
        lambda_context=args.lambda_context,
        lambda_fusion=args.lambda_fusion
    ).to(device)

    # ===== 修改：创建伪标签生成器，启用panoptic =====
    pseudo_gen = PseudoLabelGenerator(
        num_material_classes=args.num_material_classes,
        context_channels=args.context_channels,
        material_method=args.material_method,
        use_panoptic=True,  # 使用全景分割图
        cache_materials=True
    )

    # 其余代码不变...
```

然后修改 `train_one_epoch()` 函数，支持semantic输入：

```python
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
        semantic = batch.get('semantic', None)  # 获取semantic（如果有）
        if semantic is not None:
            semantic = semantic.to(device)
        image_ids = batch['image_id']

        # ===== 修改：传入semantic给伪标签生成器 =====
        pseudo_intensity, pseudo_material, pseudo_context = pseudo_gen(
            rgb, infrared, semantic, image_ids
        )

        # Forward pass（不变）
        pred_intensity, pred_material_logits, pred_context = model(rgb)
        fused_output = fusion_module(pred_intensity, pred_material_logits, pred_context)

        # Compute loss（不变）
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

        # Backward pass（不变）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log losses（不变）
        for key in epoch_losses.keys():
            epoch_losses[key].append(loss_dict[key])

        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        if writer and batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}_loss', value, global_step)

    avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    return avg_losses
```

同样修改 `validate()` 函数。

---

### **步骤1.3: 创建配置文件**

创建 `configs/pretrain_parquet.yaml`:

```yaml
# 数据配置
train_data: "/root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet"  # 修改为您的实际路径
val_data: null  # 会自动从train_data划分
image_size: 512

# 模型配置
backbone: "resnet50"
num_material_classes: 32
context_channels: 8
fusion_hidden_dim: 64

# 损失权重
lambda_intensity: 1.0
lambda_material: 1.0
lambda_context: 0.5
lambda_fusion: 2.0

# 伪标签生成
material_method: "panoptic"  # 使用全景分割图

# 训练配置
batch_size: 8  # 根据GPU显存调整
num_epochs: 100
learning_rate: 0.0001
weight_decay: 0.00001
num_workers: 4

# 日志
checkpoint_dir: "./checkpoints/decomposition"
log_dir: "./logs/decomposition"
vis_dir: "./visualizations/decomposition"
log_interval: 50
val_interval: 1
save_interval: 10
```

---

### **步骤1.4: 启动阶段1训练**

```bash
cd physics_inspired_infrared_generation

# 测试数据加载（1分钟）
python scripts/pretrain_decomposition.py \
    --train_data /root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet \
    --batch_size 2 \
    --num_epochs 1 \
    --num_workers 0

# 如果测试通过，开始正式训练
python scripts/pretrain_decomposition.py \
    --config configs/pretrain_parquet.yaml

# 或者直接指定参数（如果不用配置文件）
python scripts/pretrain_decomposition.py \
    --train_data /root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet \
    --batch_size 8 \
    --num_epochs 100 \
    --backbone resnet50 \
    --num_material_classes 32 \
    --learning_rate 1e-4
```

**预期输出:**
```
Using device: cuda
Loading datasets from parquet...
✅ Dataloaders created successfully
Train batches: 1125
Val batches: 125
Creating model...
[INFO] MultiTaskDecompositionNet initialized:
  - Backbone: resnet50 (pretrained=True, frozen=False)
  - Projection dim: 64
  - Total parameters: 25,557,032

==================================================
Epoch 1/100
==================================================
Epoch 1: 100%|████| 1125/1125 [06:23<00:00, loss=2.3456]
Train Loss: 2.3456
Val Loss: 2.1234
Saved best model to ./checkpoints/decomposition/best_model.pth
```

---

### **步骤1.5: 监控阶段1训练**

**方法1: TensorBoard**
```bash
# 新开一个终端
conda activate physics_ir
cd physics_inspired_infrared_generation
tensorboard --logdir ./logs/decomposition --port 6006

# 浏览器打开 http://localhost:6006
```

**方法2: 查看可视化**
```bash
# 每个epoch会生成可视化
ls -lh visualizations/decomposition/

# 查看最新的可视化
# 应该看到: intensity突出亮区域, material显示分块, context显示纹理
```

**关键指标:**
- `fusion_loss` 应该逐渐下降（最重要）
- `intensity_loss` 应该下降到 < 0.1
- `material_loss` 应该下降到 < 1.0
- 可视化中intensity应该与红外图的亮区对应

---

### **步骤1.6: 验证训练结果**

训练完成后（约6-8小时），验证模型:

```bash
python -c "
import torch
from decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion

# 加载最佳模型
checkpoint = torch.load('./checkpoints/decomposition/best_model.pth')
print(f'Best epoch: {checkpoint[\"epoch\"]}')
print(f'Best val loss: {checkpoint[\"val_loss\"]:.4f}')

# 测试前向传播
model = MultiTaskDecompositionNet(backbone='resnet50', num_material_classes=32)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

rgb = torch.randn(1, 3, 512, 512).cuda()
intensity, material, context = model(rgb)

print(f'\n✅ 模型加载成功')
print(f'Intensity: {intensity.shape}, range: [{intensity.min():.3f}, {intensity.max():.3f}]')
print(f'Material: {material.shape}, unique classes: {material.argmax(dim=1).unique().numel()}')
print(f'Context: {context.shape}')
"
```

---

## 🔗 阶段2: 集成到FLUX训练

### **目标**
将预训练的分解网络集成到您原有的FLUX训练流程中。

### **步骤2.1: 准备文件**

```bash
# 复制分解模块到原项目
cp -r physics_inspired_infrared_generation/decomposition \
     ICEdit_contrastive/train/src/

# 复制cross-attention
cp physics_inspired_infrared_generation/flux_integration/cross_attention.py \
   ICEdit_contrastive/train/src/decomposition/
```

---

### **步骤2.2: 修改原项目模型**

这部分代码较长，我创建一个patch文件：

```bash
cd ICEdit_contrastive/train/src/train
```

在 `model.py` 中添加（在`__init__`方法中）:

```python
# 在 __init__ 方法的最后，self.to(device).to(dtype) 之前添加:

        # ===== 新增：分解网络集成 =====
        use_decomposition = self.model_config.get('use_decomposition_guidance', False)
        self.decomposition_net = None
        self.decomp_encoder = None
        self.decomp_cross_attn = None

        if use_decomposition:
            print('[INFO] Decomposition Guidance ENABLED')

            from ..decomposition.model import MultiTaskDecompositionNet, PhysicsInspiredFusion
            from ..decomposition.cross_attention import DecompositionCrossAttention, DecompositionEncoder

            decomp_config = self.model_config.get('decomposition', {})

            # 创建分解网络
            self.decomposition_net = MultiTaskDecompositionNet(
                backbone=decomp_config.get('backbone', 'resnet50'),
                pretrained=False,
                num_material_classes=decomp_config.get('num_material_classes', 32),
                context_channels=decomp_config.get('context_channels', 8)
            )

            # 加载预训练权重
            pretrained_path = decomp_config.get('pretrained_checkpoint', None)
            if pretrained_path and os.path.exists(pretrained_path):
                print(f'[INFO] Loading pretrained decomposition from: {pretrained_path}')
                checkpoint = torch.load(pretrained_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    self.decomposition_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.decomposition_net.load_state_dict(checkpoint)
                print('[INFO] Decomposition checkpoint loaded')

            # 冻结分解网络
            freeze_decomp = decomp_config.get('freeze', True)
            if freeze_decomp:
                for param in self.decomposition_net.parameters():
                    param.requires_grad = False
                self.decomposition_net.eval()
                print('[INFO] Decomposition network is FROZEN')

            # 创建编码器
            self.decomp_encoder = DecompositionEncoder(
                num_material_classes=decomp_config.get('num_material_classes', 32),
                context_channels=decomp_config.get('context_channels', 8),
                hidden_dim=128,
                output_dim=128
            )

            # 创建交叉注意力
            self.decomp_cross_attn = DecompositionCrossAttention(
                image_dim=64,  # FLUX packed latent dimension
                decomp_dim=128,
                num_heads=8,
                dropout=0.1
            )

            print(f'[INFO] Decomposition guidance initialized')

        self.to(device).to(dtype)
```

在 `configure_optimizers()` 方法中添加:

```python
def configure_optimizers(self):
    # ... 原有代码 ...

    # 添加分解模块参数
    if self.decomp_encoder is not None:
        self.trainable_params.extend(list(self.decomp_encoder.parameters()))
        print(f'[INFO] Added decomp_encoder parameters')

    if self.decomp_cross_attn is not None:
        self.trainable_params.extend(list(self.decomp_cross_attn.parameters()))
        print(f'[INFO] Added decomp_cross_attn parameters')

    # ... 创建优化器的代码 ...
```

在 `step()` 方法中添加（在semantic processing之后）:

```python
def step(self, batch):
    # ... 原有的语义处理代码 ...

    # ===== 新增：分解网络处理 =====
    decomp_features = None
    use_decomposition = (
        self.decomposition_net is not None and
        self.model_config.get('use_decomposition_guidance', False)
    )

    if use_decomposition and hasattr(self, 'visible_img'):  # visible_img在语义处理时已定义
        from torchvision import transforms

        # ImageNet normalization for decomposition network
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        visible_normalized = normalize(visible_img.float())

        # Forward through decomposition network
        with torch.no_grad() if self.model_config.get('decomposition', {}).get('freeze', True) else torch.enable_grad():
            intensity, material_logits, context = self.decomposition_net(visible_normalized)

        # Encode decomposition features
        decomp_features = self.decomp_encoder(intensity, material_logits, context)

        # Upsample to 2048 sequence length (to match x_t)
        if decomp_features.shape[1] != 2048:
            import torch.nn.functional as F
            decomp_features = F.interpolate(
                decomp_features.transpose(1, 2),
                size=2048,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

    # ... 准备x_t等（原有代码）...

    # ===== 应用交叉注意力 =====
    # 先应用semantic guidance（如果有）
    if self.semantic_cross_attn is not None and semantic_tokens is not None:
        x_t = self.semantic_cross_attn(x_t, semantic_tokens)

    # 再应用decomposition guidance（如果有）
    if self.decomp_cross_attn is not None and decomp_features is not None:
        x_t = self.decomp_cross_attn(x_t, decomp_features)

    # 重新组装hidden_states
    hidden_states = torch.cat((x_t, x_cond), dim=2)

    # ... FLUX transformer forward（原有代码）...
```

---

### **步骤2.3: 创建配置文件**

创建 `ICEdit_contrastive/train/train/config/vis2ir_decomposition.yaml`:

```yaml
flux_path: "/root/autodl-tmp/qyt/hfd_model"
dtype: "bfloat16"

model:
  union_cond_attn: true
  add_cond_attn: false
  latent_lora: false
  use_sep: false

  # Semantic conditioning
  use_semantic_conditioning: true
  semantic_mode: "cross_attention"
  semantic_num_layers: 1
  semantic_num_heads: 8

  # ===== Decomposition guidance (新增) =====
  use_decomposition_guidance: true
  decomposition:
    backbone: "resnet50"
    num_material_classes: 32
    context_channels: 8
    # 重要：修改为阶段1训练的权重路径
    pretrained_checkpoint: "/path/to/physics_inspired_infrared_generation/checkpoints/decomposition/best_model.pth"
    freeze: true  # 冻结分解网络

train:
  batch_size: 4
  accumulate_grad_batches: 1
  dataloader_workers: 4
  save_interval: 1000
  sample_interval: 1000
  max_steps: -1
  gradient_checkpointing: true
  save_path: "runs"

  condition_type: "vis2ir_decomposition"
  dataset:
    type: "vis2ir_semantic"
    path: "/root/autodl-tmp/qyt_1/dataset/pid_llvip_dataset.parquet"
    condition_size: 512
    target_size: 512
    image_size: 512
    padding: 8
    drop_text_prob: 0.1
    drop_image_prob: 0.0
    include_semantic: true
    semantic_column: panoptic_img

  wandb:
    project: "Vis2IR-Decomposition"

  lora_config:
    r: 32
    lora_alpha: 32
    init_lora_weights: "gaussian"
    target_modules: "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"

  optimizer:
    type: "Prodigy"
    params:
      lr: 1
      use_bias_correction: true
      safeguard_warmup: true
      weight_decay: 0.01
```

---

### **步骤2.4: 启动阶段2训练**

```bash
cd ICEdit_contrastive/train

# 重要：修改配置文件中的checkpoint路径
# 编辑 train/config/vis2ir_decomposition.yaml
# 将 pretrained_checkpoint 改为实际路径

# 启动训练
python train/train.py \
    --config train/config/vis2ir_decomposition.yaml \
    --devices 0,1  # 使用的GPU ID
```

**预期输出:**
```
[INFO] Decomposition Guidance ENABLED
[INFO] Loading pretrained decomposition from: /path/to/best_model.pth
[INFO] Decomposition checkpoint loaded
[INFO] Decomposition network is FROZEN
[INFO] Decomposition guidance initialized
[INFO] Added decomp_encoder parameters
[INFO] Added decomp_cross_attn parameters
Trainable parameters: 45,123,456

Starting training...
Epoch 0: 100%|████| 2500/2500 [2:15:30<00:00, loss=0.0234]
```

---

### **步骤2.5: 监控阶段2训练**

```bash
# TensorBoard
tensorboard --logdir runs/ --port 6007

# WandB (如果配置了)
# 访问 https://wandb.ai/your-project

# 查看生成样本
ls runs/*/samples/
```

---

## 🎓 阶段3: 端到端微调（可选）

### **步骤3.1: 修改配置**

```yaml
# vis2ir_decomposition.yaml
model:
  decomposition:
    freeze: false  # 解冻分解网络

train:
  optimizer:
    params:
      lr: 0.5  # 降低学习率
```

### **步骤3.2: 从checkpoint继续训练**

```bash
python train/train.py \
    --config train/config/vis2ir_decomposition.yaml \
    --resume_from_checkpoint runs/your_checkpoint.ckpt \
    --devices 0,1
```

---

## 📊 完整时间线

| 阶段 | 时间 | GPU | 输出 |
|------|------|-----|------|
| 环境准备 | 1-2小时 | - | 依赖安装完成 |
| 阶段1 | 6-8小时 | 1x3090 | best_model.pth |
| 阶段2 | 2-3天 | 2xA100 | FLUX LoRA weights |
| 阶段3 | 12小时 | 2xA100 | 微调权重 |

---

## ✅ 检查清单

启动前确认:
- [ ] Conda环境创建并激活
- [ ] PyTorch和CUDA正确安装
- [ ] HuggingFace已登录
- [ ] Parquet数据集可访问
- [ ] 数据包含 panoptic_img 列
- [ ] GPU显存充足（阶段1: 24GB, 阶段2: 40GB+）
- [ ] 磁盘空间足够（至少100GB）

---

需要我补充任何部分吗？
