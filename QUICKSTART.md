# 🚀 快速启动指南

**完整版B流程**: 独立预训练分解网络，再集成到ICEdit

---

## 📋 前提条件

✅ 已完成ICEdit_contrastive环境配置
✅ 有parquet格式的数据集（包含src_img, edited_img, panoptic_img）
✅ GPU内存至少12GB（推荐24GB+）

---

## 步骤1: 创建数据集划分

这一步确保阶段1和阶段2使用相同的训练/验证集划分。

```bash
cd physics_inspired_infrared_generation

# 设置缓存路径（与ICEdit一致，避免重复下载）
export HF_DATASETS_CACHE="/root/autodl-tmp/.cache"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/.cache"

# 创建数据集划分
python utils/fixed_split.py \
    --parquet /root/autodl-tmp/qyt_contrasctivew/lmr_contrastive/dataset/pid_llvip_dataset_fixed.parquet \
    --output_dir ./data \
    --verify
```

**预期输出：**
```
📂 读取数据集元数据: /root/autodl-tmp/.../pid_llvip_dataset_fixed.parquet
   总样本数: XXXX
🔀 划分数据集 (ratio=0.9, seed=42)...
   训练集: XXXX 样本
   验证集: XXXX 样本
✅ 划分信息已保存到: ./data/dataset_split.json
✅ 数据集划分一致性验证通过!
```

---

## 步骤2: 修改配置文件

编辑 `configs/pretrain_decomposition.yaml`：

```yaml
data:
  # 修改为你的parquet路径
  parquet_path: "/root/autodl-tmp/qyt_contrasctivew/lmr_contrastive/dataset/pid_llvip_dataset_fixed.parquet"

  # 保持默认
  split_file: "./data/dataset_split.json"
  batch_size: 8  # 根据GPU内存调整：24GB=8, 12GB=4
  num_workers: 4
  image_size: 512
```

---

## 步骤3: 开始训练

```bash
# 方法1: 使用bash脚本（推荐）
bash scripts/train_stage1.sh

# 方法2: 直接运行Python
python scripts/pretrain_decomposition.py --config configs/pretrain_decomposition.yaml
```

**训练过程：**
```
======================================
开始训练分解网络
======================================
🏗️  创建分解网络...
   参数量: XX.XXM
📂 加载数据集: ...
✅ 数据加载器创建完成:
   训练集: XXXX 样本, XXX 批次
   验证集: XXXX 样本, XXX 批次

Epoch 0/50: 100%|██████████| XXX/XXX [XX:XX<00:00,  X.XXit/s, loss=X.XXXX, fusion=X.XXXX]

Epoch 0/50
  Train Loss: X.XXXX (Int: X.XXXX, Mat: X.XXXX, Ctx: X.XXXX, Fus: X.XXXX)
  Val Loss:   X.XXXX (Int: X.XXXX, Mat: X.XXXX, Ctx: X.XXXX, Fus: X.XXXX)
   ⭐ 最佳模型已保存: ./checkpoints/decomposition/best_model.pth
```

---

## 步骤4: 监控训练

```bash
# 在另一个终端运行TensorBoard
tensorboard --logdir ./checkpoints/decomposition/logs/

# 打开浏览器访问: http://localhost:6006
```

**关注指标：**
- `val/fusion_loss` - **最重要！** 重建红外图的质量
- `val/total_loss` - 总损失
- `train/lr` - 学习率曲线

---

## 步骤5: 检查结果

训练完成后，检查生成的文件：

```bash
ls -lh checkpoints/decomposition/

# 应该看到:
# best_model.pth    - 最佳模型
# final_model.pth   - 最后一个epoch的模型
# epoch_*.pth       - 定期保存的检查点
# logs/             - TensorBoard日志
# visualizations/   - 可视化结果（如果启用）
```

---

## 常见问题

### Q1: Out of Memory (OOM)

**解决方法：**
```yaml
# 修改 configs/pretrain_decomposition.yaml
data:
  batch_size: 4  # 或更小
  image_size: 256  # 降低分辨率
```

### Q2: 训练很慢

**原因：** `num_workers=0` 或 硬盘IO慢

**解决方法：**
```yaml
data:
  num_workers: 4  # 增加到4-8
```

### Q3: 数据集划分失败 (Killed)

**原因：** 内存不足

**已修复：** 现在使用PyArrow只读元数据，不会OOM

---

## 下一步

训练完成后，进入**阶段2**：集成到ICEdit_contrastive

参考 `TRAINING_WORKFLOW.md` 的阶段2部分。

---

## 完整命令总结

```bash
# 1. 设置环境
export HF_DATASETS_CACHE="/root/autodl-tmp/.cache"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/.cache"

# 2. 创建数据集划分
python utils/fixed_split.py \
    --parquet /path/to/your/data.parquet \
    --verify

# 3. 开始训练
bash scripts/train_stage1.sh

# 4. 监控（另一个终端）
tensorboard --logdir ./checkpoints/decomposition/logs/
```

---

**预计训练时间：**
- ResNet50 + 512x512: ~6-8小时（50 epochs, 单张24GB GPU）
- ResNet18 + 256x256: ~2-3小时（50 epochs, 单张12GB GPU）

**成功标志：**
- `val/fusion_loss < 0.1` (重建质量良好)
- `best_model.pth` 文件生成
- TensorBoard损失曲线收敛

---

需要帮助？查看 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) 获取详细说明。
