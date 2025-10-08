# Physics-Inspired Infrared Generation

**Physics-Inspired Multi-Task Decomposition for Infrared Image Generation**

This project implements Scheme D' from the thermal integration analysis: a physics-inspired approach to infrared image generation using multi-task decomposition and FLUX diffusion model.

> 📚 **详细技术文档**: 查看 [docs/](docs/) 目录获取完整的设计文档和理论分析
> - [方案D'详细说明](docs/PLAN_D_PRIME_DETAILED_EXPLANATION.md)
> - [实施指南](docs/PLAN_D_PRIME_IMPLEMENTATION_GUIDE.md)
> - [可行性分析](docs/PHYSICS_DISENTANGLEMENT_FEASIBILITY.md)

---

## 📋 Overview

Traditional semantic segmentation provides "what it is" but not "how hot". This project addresses high-temperature region chaos in infrared generation by learning physics-inspired representations that RESEMBLE thermal properties.

### Key Concept

**NOT Physical Decomposition** (impossible from RGB alone)
**BUT Physics-Inspired Learning** (using physics as inductive bias)

We decompose RGB images into three task-relevant representations:
- **Intensity**: Brightness distribution resembling temperature (high = hot regions)
- **Material**: 32 material classes resembling different thermal properties
- **Context**: 8-channel environmental features

These are learned from data, not computed from true physics, and guide FLUX diffusion model to generate realistic infrared images.

---

## 🏗️ Architecture

```
RGB Image
    ↓
[Multi-Task Decomposition Network]
    ├── Intensity Head    → [B, 1, H, W]
    ├── Material Head     → [B, 32, H, W] (32 classes)
    └── Context Head      → [B, 8, H, W]
    ↓
[Physics-Inspired Fusion]
    ↓
[Decomposition Encoder]
    ↓
[FLUX Diffusion Model + LoRA]
    ├── Cross-Attention (inject decomposition features)
    └── Transformer Blocks
    ↓
Generated Infrared Image
```

---

## 📁 Project Structure

```
physics_inspired_infrared_generation/
├── decomposition/              # Multi-task decomposition network
│   ├── model.py               # Network architecture
│   ├── losses.py              # Multi-task loss functions
│   └── pseudo_labels.py       # Pseudo-label generation
│
├── flux_integration/           # FLUX integration
│   ├── model.py               # Decomposition-guided FLUX model
│   └── cross_attention.py     # Cross-attention for guidance
│
├── utils/                      # Utilities
│   ├── data_utils.py          # Dataset and dataloaders
│   └── visualization.py       # Visualization tools
│
├── scripts/                    # Training scripts
│   ├── pretrain_decomposition.py    # Stage 1: Pretrain decomposition
│   └── train_flux_integration.py    # Stage 2: Train FLUX integration
│
├── configs/                    # Configuration files
│   ├── pretrain_decomposition.yaml
│   └── train_flux_integration.yaml
│
└── data/                       # Data directory (create this)
    ├── train/
    │   ├── rgb/
    │   └── infrared/
    └── val/
        ├── rgb/
        └── infrared/
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n physics_ir python=3.10
conda activate physics_ir

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install diffusers transformers accelerate peft
pip install tensorboard matplotlib scikit-learn pyyaml tqdm
pip install pillow numpy opencv-python datasets
```

### 1.5. Configure Cache (Recommended)

**Important**: Set cache paths to avoid repeated downloads and save disk space.

```bash
# Set HuggingFace cache location (same as ICEdit_contrastive)
export HF_DATASETS_CACHE="/path/to/your/cache"  # e.g., "/root/autodl-tmp/.cache"
export HUGGINGFACE_HUB_CACHE="/path/to/your/cache"

# Add to ~/.bashrc for persistence
echo 'export HF_DATASETS_CACHE="/path/to/your/cache"' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE="/path/to/your/cache"' >> ~/.bashrc
source ~/.bashrc
```

This ensures:
- No duplicate dataset downloads between Stage 1 and Stage 2
- Consistent with ICEdit_contrastive caching strategy
- Saves significant disk space

### 2. Prepare Data

Organize your RGB-Infrared paired data:

```
data/
├── train/
│   ├── rgb/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── infrared/
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
└── val/
    ├── rgb/
    └── infrared/
```

**Note**: RGB and infrared images must have matching filenames.

### 3. Stage 1: Pretrain Decomposition Network

```bash
python scripts/pretrain_decomposition.py \
    --config configs/pretrain_decomposition.yaml
```

Or with custom arguments:

```bash
python scripts/pretrain_decomposition.py \
    --train_data ./data/train \
    --val_data ./data/val \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --backbone resnet50
```

**Monitor training:**
```bash
tensorboard --logdir ./logs/decomposition
```

**What this does:**
- Trains decomposition network with pseudo-labels
- Learns intensity, material, context representations
- Saves checkpoints to `./checkpoints/decomposition/`
- Best model: `./checkpoints/decomposition/best_model.pth`

### 4. Stage 2: Train FLUX Integration

**Important**: Requires access to FLUX.1-Fill-dev (authenticate with HuggingFace)

```bash
# Login to HuggingFace (first time only)
huggingface-cli login

# Train FLUX integration
python scripts/train_flux_integration.py \
    --config configs/train_flux_integration.yaml \
    --decomposition_checkpoint ./checkpoints/decomposition/best_model.pth
```

**What this does:**
- Loads pretrained decomposition network (frozen)
- Applies LoRA to FLUX transformer
- Trains cross-attention to inject decomposition guidance
- Saves checkpoints to `./checkpoints/flux/`

**Memory Requirements:**
- Minimum: 24GB GPU (RTX 3090/4090) with `batch_size=1`
- Recommended: 40GB+ GPU (A100) with `batch_size=4`

---

## 📊 Monitoring Training

### TensorBoard

```bash
# Decomposition pretraining
tensorboard --logdir ./logs/decomposition

# FLUX integration
tensorboard --logdir ./logs/flux
```

### Visualizations

- Decomposition results: `./visualizations/decomposition/`
- Generated infrared: `./visualizations/flux/`

---

## 🎯 Training Tips

### Stage 1 (Decomposition Pretraining)

1. **Start with `material_method='simple'`** for faster iteration
2. **Switch to `material_method='kmeans'`** for better quality (slower)
3. **Monitor fusion reconstruction loss** - most important metric
4. **Check visualizations** - ensure intensity captures hot regions
5. **Train for 50-100 epochs** until convergence

### Stage 2 (FLUX Integration)

1. **Start with frozen decomposition** (`freeze_decomposition=true`)
2. **Use lower learning rate** (1e-5) for stability
3. **Monitor generated samples** frequently - visual quality matters
4. **Adjust LoRA rank** if needed (higher = more expressive, more memory)
5. **Optional Stage 3**: Unfreeze and joint fine-tune after convergence

---

## ⚙️ Configuration

### Key Parameters

**Decomposition Network:**
- `backbone`: ResNet backbone (resnet18 or resnet50)
- `num_material_classes`: Number of material types (default: 32)
- `context_channels`: Context feature dimension (default: 8)
- `lambda_fusion`: Most important loss weight (default: 2.0)

**FLUX Integration:**
- `lora_rank`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha (default: 32)
- `freeze_decomposition`: Freeze decomposition during Stage 2 (default: true)
- `learning_rate`: Lower than Stage 1 (default: 1e-5)

### Adjusting for Your GPU

**24GB GPU (RTX 3090/4090):**
```yaml
# pretrain_decomposition.yaml
batch_size: 4

# train_flux_integration.yaml
batch_size: 1
```

**16GB GPU (RTX 4080):**
```yaml
# pretrain_decomposition.yaml
batch_size: 2
image_size: 256

# train_flux_integration.yaml
batch_size: 1
image_size: 256
lora_rank: 8  # Reduce rank
```

---

## 📈 Expected Results

### Stage 1 (Decomposition)
- **Intensity**: Should capture bright regions in infrared
- **Material**: Should segment into coherent material regions
- **Context**: Should capture environmental features
- **Fusion**: Should reconstruct infrared reasonably well

### Stage 2 (FLUX Integration)
- **Generation Quality**: Should improve over baseline FLUX
- **High-Temperature Regions**: Should be more coherent and realistic
- **Physical Plausibility**: Should respect material properties

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size`
2. Reduce `image_size`
3. Use gradient checkpointing (add to model)
4. Reduce `lora_rank`

### Poor Decomposition Quality

1. Train longer (more epochs)
2. Use `material_method='kmeans'` instead of 'simple'
3. Adjust loss weights (increase `lambda_fusion`)
4. Check data quality

### FLUX Training Unstable

1. Lower `learning_rate` (try 5e-6)
2. Check decomposition checkpoint is loaded correctly
3. Ensure decomposition is frozen (`freeze_decomposition=true`)
4. Monitor gradient norms

### Material Labels Too Noisy

1. Use `material_method='kmeans'` for better clustering
2. Increase `num_material_classes` (e.g., 64)
3. Enable material caching for consistency

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{physics_inspired_ir,
  title={Physics-Inspired Multi-Task Decomposition for Infrared Image Generation},
  author={Your Name},
  year={2025}
}
```

---

## ⚠️ Important Notes

### This is NOT Physics Disentanglement

❌ **We do NOT claim to:**
- Extract true temperature from RGB
- Compute true emissivity values
- Perform physical decomposition

✅ **We DO:**
- Learn representations that RESEMBLE physics quantities
- Use physics as inductive bias for better learning
- Generate more realistic infrared images

### Mathematical Justification

**Why can't we do true physics decomposition?**

From Stefan-Boltzmann law: `I = ε × σ × T^4 + (1-ε) × X`

- 1 equation (infrared intensity)
- 3 unknowns (T, ε, X)
- **Underconstrained problem** = no unique solution
- RGB contains visible light, not thermal information

**What we do instead:**

Learn task-relevant approximations that are GOOD ENOUGH for guiding diffusion models.

---

## 🛠️ Future Work

1. **Implement FLUX forward pass hooks** (currently placeholder)
2. **Add gradient checkpointing** for memory efficiency
3. **Experiment with different fusion strategies**
4. **Try different pseudo-label generation methods**
5. **Add perceptual loss** for better visual quality
6. **Implement Stage 3** (joint fine-tuning)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 📜 License

[Add your license here]

---

## 🙏 Acknowledgments

- **HADAR Project**: Inspiration for physics-based thermal decomposition
- **FLUX**: Black Forest Labs for the diffusion model
- **Diffusers**: HuggingFace for the excellent library

---

**Happy Training! 🚀**
"# piig" 
