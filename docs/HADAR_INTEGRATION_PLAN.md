# HADAR + Contrastive Learning 集成方案

## 核心思路

用 HADAR 的 TeX 分解替代全景语义分割，提供物理约束的热量信息。

---

## 问题分析

### 当前问题
红外图像中的**高温区域（高亮区域）混乱、不准确**

### 根本原因
1. **语义分割只提供"是什么"，不提供"多热"**
   - 全景分割: 汽车、道路、建筑 → 类别信息
   - 缺失信息: 发动机350K、车身300K、轮胎280K → 温度信息

2. **分割错误累积**
   - 语义分割本身不是完美GT
   - 错误的类别 → 错误的红外生成

3. **缺乏物理约束**
   - 当前方案纯数据驱动
   - 生成的红外图可能违反热辐射物理定律

### HADAR解决方案
通过**TeX分解**提供物理准确的热量信息：
- **T (Temperature)**: 直接预测温度 → 解决"多热"问题
- **ε (Emissivity)**: 材质发射率 → 提供物理约束
- **X (Texture)**: 环境反射 → 捕捉细节纹理

---

## 方案1: TeX-Guided Generation (推荐 - 最小改动)

### 1.1 架构流程

```
RGB Image (可见光)
    ↓
[TeXNet Predictor]  ← 从RGB预测温度图
    ↓
T_map (Temperature) [H, W, 1]
    ↓
[Tri-Modal Encoder] (修改第三模态)
    ↓
z_v (visible) ←→ z_ir (infrared) ←→ z_T (temperature)
    ↓                                      ↓
[Cross-Attention with T-guidance]         |
    ↓                                      |
[FLUX Generator] ← 温度引导 ──────────────┘
    ↓
Infrared Image + Physics Loss
```

### 1.2 实现步骤

#### Step 1: 创建温度预测器

**选项A: 简单伪标签方法 (快速验证)**

```python
# utils/temperature_pseudo_label.py
import torch
import torch.nn.functional as F

def extract_temperature_from_infrared(infrared_img):
    """
    从红外图像估计温度图（伪标签）

    物理依据: Stefan-Boltzmann定律 I ∝ T^4

    Args:
        infrared_img: [B, 3, H, W] 红外图像 (0-255或归一化)
    Returns:
        T_map: [B, 1, H, W] 温度图 (273-323K范围)
    """
    # 转为灰度 (红外图通常是单通道的伪彩色)
    if infrared_img.shape[1] == 3:
        gray_ir = infrared_img.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    else:
        gray_ir = infrared_img

    # 归一化到 [0, 1]
    I_min = gray_ir.view(gray_ir.size(0), -1).min(dim=1, keepdim=True)[0]
    I_max = gray_ir.view(gray_ir.size(0), -1).max(dim=1, keepdim=True)[0]
    I_min = I_min.view(-1, 1, 1, 1)
    I_max = I_max.view(-1, 1, 1, 1)

    I_norm = (gray_ir - I_min) / (I_max - I_min + 1e-8)

    # 温度估计: T = T_min + (T_max - T_min) × I^(1/4)
    # 室外场景典型温度范围: 273K (0°C) ~ 323K (50°C)
    T_min, T_max = 273.0, 323.0
    T_estimated = T_min + (T_max - T_min) * torch.pow(I_norm, 0.25)

    return T_estimated

# 使用示例
# T_pseudo = extract_temperature_from_infrared(infrared_gt)
```

**选项B: 训练专用温度预测器 (更准确)**

```python
# models/tex_predictor.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class TeXPredictor(nn.Module):
    """从RGB图像预测温度图"""

    def __init__(self,
                 encoder_name='resnet50',
                 encoder_weights='imagenet',
                 in_channels=3,
                 predict_T=True,
                 predict_epsilon=False):
        super().__init__()

        self.predict_T = predict_T
        self.predict_epsilon = predict_epsilon

        # 输出通道
        out_channels = 0
        if predict_T:
            out_channels += 1  # Temperature
        if predict_epsilon:
            out_channels += 30  # Emissivity (30 material classes)

        # 使用PAN架构 (与HADAR一致)
        self.texnet = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )

        # 温度归一化参数 (与HADAR一致)
        self.T_mu = 15.997467357494212   # 均值 (°C)
        self.T_std = 8.544861474951992   # 标准差

    def forward(self, rgb):
        """
        Args:
            rgb: [B, 3, H, W] RGB图像
        Returns:
            T_map: [B, 1, H, W] 归一化温度图
            epsilon_map: [B, 30, H, W] 发射率logits (可选)
        """
        output = self.texnet(rgb)

        results = {}
        idx = 0

        if self.predict_T:
            T_norm = output[:, idx:idx+1, :, :]  # [B, 1, H, W]
            # 温度范围约束: tanh → [-1, 1] → 合理温度范围
            T_norm = torch.tanh(T_norm)
            results['T_norm'] = T_norm

            # 反归一化到开尔文
            T_kelvin = T_norm * self.T_std + self.T_mu + 273.15
            results['T_kelvin'] = T_kelvin
            idx += 1

        if self.predict_epsilon:
            epsilon_logits = output[:, idx:idx+30, :, :]
            results['epsilon_logits'] = epsilon_logits

        return results

# 训练损失
class TemperaturePredictorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, T_pred, T_gt):
        """
        Args:
            T_pred: [B, 1, H, W] 预测温度 (归一化)
            T_gt: [B, 1, H, W] 真值温度 (归一化)
        """
        loss = self.mse_loss(T_pred, T_gt)
        return loss
```

**训练脚本**:
```python
# train_tex_predictor.py
import torch
from torch.utils.data import DataLoader
from models.tex_predictor import TeXPredictor, TemperaturePredictorLoss
from utils.temperature_pseudo_label import extract_temperature_from_infrared

# 初始化
model = TeXPredictor(predict_T=True).cuda()
criterion = TemperaturePredictorLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(50):
    for batch in dataloader:
        rgb = batch['visible'].cuda()
        infrared = batch['infrared'].cuda()

        # 生成伪标签
        T_pseudo = extract_temperature_from_infrared(infrared)

        # 归一化到与HADAR一致
        T_mu, T_std = 15.997467, 8.544861
        T_pseudo_celsius = T_pseudo - 273.15
        T_pseudo_norm = (T_pseudo_celsius - T_mu) / T_std

        # 前向
        results = model(rgb)
        T_pred = results['T_norm']

        # 损失
        loss = criterion(T_pred, T_pseudo_norm)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Step 2: 修改 Tri-Modal Encoder

**修改 `models/tri_encoder.py`**:

```python
class TriModalEncoder(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()

        # 三个分支编码器
        self.encoder_visible = self._make_encoder(backbone, in_channels=3)
        self.encoder_infrared = self._make_encoder(backbone, in_channels=3)
        self.encoder_thermal = self._make_encoder(backbone, in_channels=1)  # ← 改为1通道温度图

    def forward(self, visible, infrared, thermal_map):
        """
        Args:
            visible: [B, 3, H, W] 可见光
            infrared: [B, 3, H, W] 红外
            thermal_map: [B, 1, H, W] 温度图 (新!)
        Returns:
            z_v, z_ir, z_T: [B, 512] 特征向量
        """
        z_v = self.encoder_visible(visible)
        z_ir = self.encoder_infrared(infrared)
        z_T = self.encoder_thermal(thermal_map)  # ← 温度特征

        return z_v, z_ir, z_T
```

**修改训练脚本 `train/scripts/pretrain_encoder.py`**:

```python
# 旧版本
# z_v, z_ir, z_s = tri_encoder(visible, infrared, semantic)

# 新版本
from utils.temperature_pseudo_label import extract_temperature_from_infrared

# 在每个batch中
T_map = extract_temperature_from_infrared(infrared)  # [B, 1, H, W]

# 或者使用预训练的预测器
# tex_predictor = load_tex_predictor('checkpoints/tex_predictor.pth')
# with torch.no_grad():
#     results = tex_predictor(visible)
#     T_map = results['T_kelvin']

z_v, z_ir, z_T = tri_encoder(visible, infrared, T_map)

# 对比损失保持不变
loss_contrast = contrastive_loss(z_v, z_ir, z_T)
```

#### Step 3: 在FLUX生成中添加温度引导

**修改 `models/icedit_flux_model.py`**:

```python
class ICEditFLUXModel(nn.Module):
    def forward(self, visible, infrared=None, semantic=None, temperature=None):
        """
        Args:
            temperature: [B, 1, H, W] 温度图 (新增)
        """
        # 编码温度信息
        if temperature is not None:
            # 温度嵌入 (映射到FLUX的条件空间)
            temp_tokens = self.temperature_encoder(temperature)  # [B, N, D]

            # 与可见光特征融合
            visible_tokens = self.vision_encoder(visible)

            # Cross-attention: 温度引导可见光
            guided_tokens = self.cross_attn(visible_tokens, temp_tokens)
        else:
            guided_tokens = self.vision_encoder(visible)

        # FLUX生成
        latent_ir = self.flux_model(guided_tokens, ...)

        return latent_ir
```

**添加温度编码器**:
```python
class TemperatureEncoder(nn.Module):
    """将温度图编码为tokens"""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, 2, 1),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim))

    def forward(self, T_map):
        # T_map: [B, 1, H, W]
        x = self.conv(T_map)  # [B, D, H', W']
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', D]
        x = x + self.pos_embed[:, :x.size(1)]
        return x
```

#### Step 4: 添加物理约束损失

```python
# losses/physics_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalPhysicsLoss(nn.Module):
    """基于物理定律的约束损失"""

    def __init__(self, lambda_stefan=1.0, lambda_smooth=0.1, lambda_range=0.01):
        super().__init__()
        self.lambda_stefan = lambda_stefan
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range

    def stefan_boltzmann_loss(self, infrared_img, T_map):
        """
        Stefan-Boltzmann定律: I ∝ ε × σ × T^4
        简化: I ∝ T^4 (假设发射率恒定)

        Args:
            infrared_img: [B, 3, H, W] 生成的红外图
            T_map: [B, 1, H, W] 温度图 (开尔文)
        """
        # 计算预期辐射强度
        expected_radiance = torch.pow(T_map / 300.0, 4)  # 归一化到300K

        # 计算实际辐射强度 (红外图的亮度)
        actual_radiance = infrared_img.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 归一化
        actual_radiance = (actual_radiance - actual_radiance.min()) / \
                          (actual_radiance.max() - actual_radiance.min() + 1e-8)
        expected_radiance = (expected_radiance - expected_radiance.min()) / \
                            (expected_radiance.max() - expected_radiance.min() + 1e-8)

        # MSE损失
        loss = F.mse_loss(actual_radiance, expected_radiance)
        return loss

    def temperature_smoothness_loss(self, T_map):
        """
        温度平滑约束: 相邻像素温度应该连续
        Total Variation Loss
        """
        # 水平方向梯度
        diff_h = torch.abs(T_map[:, :, 1:, :] - T_map[:, :, :-1, :])
        # 垂直方向梯度
        diff_v = torch.abs(T_map[:, :, :, 1:] - T_map[:, :, :, :-1])

        loss = diff_h.mean() + diff_v.mean()
        return loss

    def temperature_range_loss(self, T_map):
        """
        温度范围约束: 温度应在合理物理范围内
        室外场景: 250K (-23°C) ~ 350K (77°C)
        """
        T_min, T_max = 250.0, 350.0

        # 惩罚超出范围的温度
        loss_lower = F.relu(T_min - T_map).mean()
        loss_upper = F.relu(T_map - T_max).mean()

        loss = loss_lower + loss_upper
        return loss

    def forward(self, infrared_img, T_map):
        """
        Args:
            infrared_img: [B, 3, H, W] 生成的红外图
            T_map: [B, 1, H, W] 温度图 (开尔文)
        """
        loss_stefan = self.stefan_boltzmann_loss(infrared_img, T_map)
        loss_smooth = self.temperature_smoothness_loss(T_map)
        loss_range = self.temperature_range_loss(T_map)

        total_loss = (self.lambda_stefan * loss_stefan +
                      self.lambda_smooth * loss_smooth +
                      self.lambda_range * loss_range)

        return {
            'loss_physics': total_loss,
            'loss_stefan': loss_stefan,
            'loss_smooth': loss_smooth,
            'loss_range': loss_range
        }
```

**集成到训练中**:
```python
# 在 train_joint.py 中
physics_loss_fn = ThermalPhysicsLoss(
    lambda_stefan=1.0,
    lambda_smooth=0.1,
    lambda_range=0.01
)

# 训练循环
for batch in dataloader:
    # 生成红外图
    infrared_pred = model(visible, temperature=T_map)

    # 原有损失
    loss_recon = F.mse_loss(infrared_pred, infrared_gt)
    loss_contrast = contrastive_loss(...)

    # 物理损失
    physics_losses = physics_loss_fn(infrared_pred, T_map)
    loss_physics = physics_losses['loss_physics']

    # 总损失
    total_loss = loss_recon + 0.1 * loss_contrast + 0.05 * loss_physics
```

---

## 方案2: 完整TeX分解 (进阶 - 更强物理约束)

### 2.1 架构

```
RGB Image
    ↓
[TeXNet] ← 从HADAR预训练
    ↓
T (Temperature) [H, W, 1]
ε (Emissivity)  [H, W, 30]
X (Texture)     [H, W, C]
    ↓
[Physical Synthesis] ← 普朗克定律
    ↓
I_physics = ε × B(T) + (1-ε) × X
    ↓
[Tri-Modal Encoder]
z_v ←→ z_physics ←→ z_ir
    ↓
[FLUX Generator]
    ↓
Infrared Image
```

### 2.2 完整物理合成

```python
# models/full_tex_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullTeXGenerator(nn.Module):
    """完整TeX分解 + 物理合成"""

    def __init__(self):
        super().__init__()

        # TeX预测网络 (可以加载HADAR预训练权重)
        from models.tex_predictor import TeXPredictor
        self.texnet = TeXPredictor(
            predict_T=True,
            predict_epsilon=True
        )

        # 发射率材料库 (从HADAR加载)
        self.register_buffer('emissivity_lib', self.load_emissivity_library())

        # FLUX生成器
        self.flux_model = ICEditFLUXModel()

    def load_emissivity_library(self):
        """
        加载30种材质的发射率数据
        返回: [30, 49] 发射率光谱
        """
        import numpy as np
        # 从HADAR加载 (示例路径)
        # emi_path = 'HADAR/TeXNet/emissivity_library.npy'
        # emi_val = np.load(emi_path)

        # 临时: 使用简化的单波段发射率
        emi_val = torch.tensor([
            0.95,  # 0: 沥青
            0.92,  # 1: 混凝土
            0.25,  # 2: 金属
            0.85,  # 3: 玻璃
            0.98,  # 4: 植被
            0.96,  # 5: 水
            # ... 共30种材质
        ])
        # 扩展到多波段
        emi_val = emi_val.unsqueeze(1).repeat(1, 49)  # [30, 49]
        return emi_val

    def planck_blackbody(self, T, wavelength=10e-6):
        """
        普朗克黑体辐射定律

        Args:
            T: [B, 1, H, W] 温度 (开尔文)
            wavelength: 波长 (米), 默认10μm (长波红外)
        Returns:
            B_lambda: [B, 1, H, W] 辐射亮度
        """
        h = 6.62607015e-34  # 普朗克常数
        c = 2.99792458e8    # 光速
        k = 1.380649e-23    # 玻尔兹曼常数

        # 防止数值溢出
        exp_term = h * c / (wavelength * k * T)
        exp_term = torch.clamp(exp_term, max=50.0)

        numerator = 2 * h * c**2
        denominator = wavelength**5 * (torch.exp(exp_term) - 1)

        B_lambda = numerator / denominator

        # 归一化到可见范围
        B_lambda = B_lambda / 1e6  # 简化
        return B_lambda

    def physical_synthesis(self, T, epsilon_logits, X=None):
        """
        物理公式: I = ε × B(T) + (1-ε) × X

        Args:
            T: [B, 1, H, W] 温度 (开尔文)
            epsilon_logits: [B, 30, H, W] 材质分类logits
            X: [B, C, H, W] 环境纹理 (可选)
        Returns:
            I_physics: [B, 3, H, W] 合成的红外辐射
        """
        B, _, H, W = T.shape

        # 1. 获取材质类别
        epsilon_map = torch.argmax(epsilon_logits, dim=1)  # [B, H, W]

        # 2. 查询发射率
        emissivity = self.emissivity_lib[epsilon_map]  # [B, H, W, 49]
        # 简化: 取平均发射率
        epsilon_val = emissivity.mean(dim=-1, keepdim=True)  # [B, H, W, 1]
        epsilon_val = epsilon_val.permute(0, 3, 1, 2)  # [B, 1, H, W]

        # 3. 计算黑体辐射
        B_T = self.planck_blackbody(T)  # [B, 1, H, W]

        # 4. 环境纹理 (如果没有提供，假设为0)
        if X is None:
            X = torch.zeros_like(B_T).repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # 5. 物理合成
        # I = ε × B(T) + (1-ε) × X
        emitted = epsilon_val * B_T  # [B, 1, H, W]
        reflected = (1 - epsilon_val) * X.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        I_physics = emitted + reflected  # [B, 1, H, W]
        I_physics = I_physics.repeat(1, 3, 1, 1)  # 扩展到3通道

        return I_physics

    def forward(self, visible_img):
        """
        完整前向流程

        Args:
            visible_img: [B, 3, H, W]
        Returns:
            infrared_generated: [B, 3, H, W]
            T_map: [B, 1, H, W]
            epsilon_map: [B, 30, H, W]
        """
        # 1. TeX预测
        tex_results = self.texnet(visible_img)
        T_map = tex_results['T_kelvin']  # [B, 1, H, W]
        epsilon_logits = tex_results['epsilon_logits']  # [B, 30, H, W]

        # 2. 物理合成
        I_physics = self.physical_synthesis(T_map, epsilon_logits)

        # 3. 用物理合成指导FLUX生成
        infrared_generated = self.flux_model(
            visible=visible_img,
            physics_prior=I_physics,
            temperature=T_map
        )

        return {
            'infrared': infrared_generated,
            'T_map': T_map,
            'epsilon_logits': epsilon_logits,
            'I_physics': I_physics
        }
```

### 2.3 损失函数

```python
# 完整训练损失
def full_tex_loss(pred, gt):
    # 1. 重建损失
    loss_recon = F.mse_loss(pred['infrared'], gt['infrared'])

    # 2. 物理一致性损失
    loss_physics_consist = F.mse_loss(pred['I_physics'], gt['infrared'])

    # 3. TeX监督损失 (如果有GT)
    loss_T = F.mse_loss(pred['T_map'], gt['T_map'])
    loss_epsilon = F.cross_entropy(pred['epsilon_logits'], gt['epsilon_map'])

    # 4. 物理约束
    physics_loss_fn = ThermalPhysicsLoss()
    loss_physics = physics_loss_fn(pred['infrared'], pred['T_map'])['loss_physics']

    # 总损失
    total = (1.0 * loss_recon +
             0.5 * loss_physics_consist +
             0.1 * loss_T +
             0.1 * loss_epsilon +
             0.05 * loss_physics)

    return total
```

---

## 数据准备

### 3.1 从现有数据集生成温度伪标签

```python
# scripts/generate_temperature_labels.py
import torch
from pathlib import Path
from tqdm import tqdm
from utils.temperature_pseudo_label import extract_temperature_from_infrared

# 加载数据集
dataset_path = Path('data/your_dataset')
output_path = Path('data/temperature_labels')
output_path.mkdir(exist_ok=True)

# 遍历所有红外图像
for infrared_file in tqdm(list(dataset_path.glob('**/infrared/*.png'))):
    # 读取红外图
    infrared = load_image(infrared_file)  # [H, W, 3]
    infrared = torch.from_numpy(infrared).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 生成温度伪标签
    T_pseudo = extract_temperature_from_infrared(infrared)  # [1, 1, H, W]

    # 保存
    save_name = infrared_file.stem + '_temperature.pt'
    torch.save(T_pseudo, output_path / save_name)

print(f"Generated {len(list(output_path.glob('*.pt')))} temperature labels")
```

### 3.2 使用HADAR数据集预训练 (可选)

如果能获取HADAR数据集：

```bash
# Step 1: 下载HADAR数据集
# https://purdue0-my.sharepoint.com/...

# Step 2: 预训练TeXNet
cd HADAR/TeXNet
python main.py \
    --data_dir /path/to/HADAR_database \
    --checkpoint_dir checkpoints/hadar_pretrained \
    --train_T \
    --train_v \
    --epochs 100

# Step 3: 转换到你的任务
# 修改输入通道: 49 (Heat Cube) → 3 (RGB)
# 方法: 冻结解码器,只微调编码器第一层
```

---

## 训练策略

### Stage 1: 预训练温度预测器 (可选)

```bash
# 使用伪标签训练
python train_tex_predictor.py \
    --data_path data/your_dataset \
    --temperature_labels data/temperature_labels \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/tex_predictor
```

### Stage 2: 对比学习 (修改版)

```bash
# 用温度图替代语义图
python train/scripts/pretrain_encoder.py \
    --use_temperature_guidance \
    --tex_predictor_path checkpoints/tex_predictor/best.pth \
    --train_parquet_path data/train.parquet \
    --output_dir checkpoints/tri_encoder_temp \
    --epochs 100
```

**关键修改**:
```python
# 在 pretrain_encoder.py 中
USE_TEMPERATURE = True

if USE_TEMPERATURE:
    # 加载温度预测器
    tex_predictor = load_tex_predictor(args.tex_predictor_path)
    tex_predictor.eval()

    # 在每个batch
    with torch.no_grad():
        T_map = tex_predictor(visible)['T_kelvin']

    # 使用温度图
    z_v, z_ir, z_T = tri_encoder(visible, infrared, T_map)
else:
    # 原来的语义图
    z_v, z_ir, z_s = tri_encoder(visible, infrared, semantic)
```

### Stage 3: 联合训练 + 物理约束

```bash
python train/scripts/train_joint.py \
    --pretrained_encoder checkpoints/tri_encoder_temp/checkpoint_final.pth \
    --train_parquet_path data/train.parquet \
    --output_dir checkpoints/joint_physics \
    --use_physics_loss \
    --lambda_physics 0.05 \
    --epochs 50
```

---

## 预期改进

### 4.1 问题 → 解决方案对照表

| 当前问题 | 根本原因 | HADAR方案 | 预期改进 |
|---------|---------|-----------|---------|
| 高温区域混乱 | 语义图只有类别，无温度信息 | 温度图T直接提供热量信息 | ✅ 高温区域准确 |
| 缺少热量细节 | 语义边界粗糙 | T、ε、X提供像素级物理量 | ✅ 细节丰富 |
| 分割错误累积 | 依赖不完美的GT分割 | 端到端预测温度 | ✅ 减少误差传播 |
| 物理不合理 | 无物理约束 | Stefan-Boltzmann定律 | ✅ 物理一致性 |
| 泛化能力弱 | 纯数据驱动 | 物理先验知识 | ✅ 更好泛化 |

### 4.2 定量指标预期

| 指标 | 当前 (语义) | 目标 (TeX) | 提升 |
|------|------------|------------|------|
| PSNR | 25.3 dB | 27.5+ dB | +2.2 dB |
| SSIM | 0.78 | 0.85+ | +0.07 |
| 高温区域PSNR | 22.1 dB | 26.0+ dB | +3.9 dB |
| 物理一致性 | N/A | 0.90+ | 新增 |

---

## 快速实验 (立即可做)

### 5.1 最简验证 (1天)

```python
# 1. 在现有代码中添加温度伪标签
# 修改 data/data.py

from utils.temperature_pseudo_label import extract_temperature_from_infrared

class YourDataset:
    def __getitem__(self, idx):
        # 原有加载
        visible = self.load_visible(idx)
        infrared = self.load_infrared(idx)
        semantic = self.load_semantic(idx)  # 旧的

        # 新增: 生成温度图
        temperature = extract_temperature_from_infrared(infrared)

        # 返回时替换
        return {
            'visible': visible,
            'infrared': infrared,
            'semantic': temperature  # ← 用温度图替换语义图!
        }

# 2. 不改任何其他代码，直接重新训练
# bash train/scripts/pretrain_encoder.sh
```

**预期结果**:
- 如果温度图有效，Stage1损失应该下降
- 高温区域应该更准确

### 5.2 进阶实验 (1周)

1. **Day 1-2**: 实现`TeXPredictor`，训练温度预测器
2. **Day 3-4**: 修改`TriModalEncoder`，集成温度分支
3. **Day 5-6**: 添加物理损失，联合训练
4. **Day 7**: 评估对比，可视化改进

---

## 代码实现优先级

### P0 (立即实现 - 今天)
- [x] `utils/temperature_pseudo_label.py` - 温度伪标签生成
- [x] 修改`data.py` - 用温度图替换语义图
- [ ] 重新训练Stage1 - 验证想法

### P1 (本周)
- [ ] `models/tex_predictor.py` - TeXPredictor网络
- [ ] `losses/physics_loss.py` - 物理约束损失
- [ ] `train_tex_predictor.py` - 训练脚本
- [ ] 修改`tri_encoder.py` - 温度分支

### P2 (下周)
- [ ] `models/full_tex_generator.py` - 完整TeX分解
- [ ] 加载HADAR预训练权重 (如果可获取)
- [ ] 集成到FLUX生成器
- [ ] 完整pipeline训练

### P3 (可选 - 论文强化)
- [ ] 多波段物理建模 (不只是单通道温度)
- [ ] 材质识别模块
- [ ] 环境反射X预测
- [ ] 无监督物理重建损失

---

## 论文创新点

### 6.1 核心贡献

1. **物理引导的红外图像生成**
   - 首次将热辐射物理定律引入可见光到红外的生成任务
   - 不是纯数据驱动，而是物理约束引导

2. **TeX分解作为中间表示**
   - Temperature (T): 物体本征温度 → 解决"多热"问题
   - Emissivity (ε): 材质特性 → 提供物理先验
   - Texture (X): 环境反射 → 捕捉细节

3. **多模态对齐的物理解释**
   - 传统: z_v ←→ z_s (可见光 ←→ 语义)
   - 本文: z_v ←→ z_T (可见光 ←→ 温度)
   - 物理量对齐比语义对齐更直接

4. **端到端可学习的物理模型**
   - 从RGB直接预测物理量 (T, ε, X)
   - 物理约束确保生成结果的合理性

### 6.2 对比相关工作

| 方法 | 引导信息 | 物理约束 | 高温区域 | 可解释性 |
|------|---------|---------|---------|---------|
| PIAFusion | 无 | ❌ | 差 | 低 |
| ICEdit (原版) | 语义分割 | ❌ | 一般 | 中 |
| **ICEdit+TeX (本文)** | **温度图** | **✅** | **优** | **高** |

### 6.3 实验设计

**消融实验**:
1. Baseline: 无引导 (只用可见光)
2. +Semantic: 加语义分割引导
3. +Temperature: 加温度图引导 (伪标签)
4. +Temperature+Physics: 加温度引导+物理损失
5. +FullTeX: 完整TeX分解 (T+ε+X)

**评估指标**:
- 传统: PSNR, SSIM, LPIPS
- 物理: Stefan-Boltzmann一致性, 温度分布相关性
- 感知: 高温区域FID, 人类评估

---

## 参考文献

### 核心论文
- **HADAR**: Hu et al., "Heat-assisted detection and ranging", *Nature*, 2024
  - [论文链接](https://www.nature.com/articles/s41586-024-07171-9)
  - 提供了TeX分解的理论和实现

### 物理基础
- **Planck's Law**: [Wikipedia](https://en.wikipedia.org/wiki/Planck%27s_law)
- **Stefan-Boltzmann Law**: [Wikipedia](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
- **Thermal Radiation**: Modest, *Radiative Heat Transfer*, 3rd Ed.

### 相关代码
- **HADAR TeXNet**: `D:\研究生论文工作\红外图像生成\HADAR\TeXNet`
- **SMP Library**: https://github.com/qubvel/segmentation_models.pytorch
- **HADAR详细文档**: `D:\研究生论文工作\红外图像生成\HADAR\HADAR_INIT.md`

---

## 总结

通过引入HADAR的TeX分解，我们将**物理知识**注入到红外图像生成中：

**核心改进**:
1. 温度图 (T) 替代语义图 → 直接提供热量信息
2. 物理约束 (Stefan-Boltzmann) → 确保合理性
3. 端到端学习 → 减少误差累积

**实施路径**:
1. 快速验证: 伪标签温度图 (1天)
2. 完整方案: 训练TeXPredictor (1周)
3. 进阶版本: 完整TeX分解 (2周)

**预期效果**: 高温区域准确度显著提升，物理一致性增强，论文创新性强。
