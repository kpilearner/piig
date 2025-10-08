#!/bin/bash

# ====================================================================
# Stage 1: Thermal Query Decoder Pretraining
# ====================================================================
#
# This script trains the ThermalQueryDecoder on RGB-Infrared pairs.
# It's much simpler than the multi-task decomposition approach!
#
# Expected training time: 6-8 hours (ResNet50, 50 epochs, 24GB GPU)
# ====================================================================

echo "========================================"
echo "Starting Stage 1 Training"
echo "Thermal Query Decoder Pretraining"
echo "========================================"

# Set cache paths (same as ICEdit to avoid duplication)
DATA_CACHE=${1:-"/root/autodl-tmp/.cache"}

if [[ -n "$DATA_CACHE" ]]; then
  export HF_DATASETS_CACHE="$DATA_CACHE"
  export HUGGINGFACE_HUB_CACHE="$DATA_CACHE"
  echo "Dataset cache path: $DATA_CACHE"
fi

# Set Python path
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Configuration file
CONFIG="configs/pretrain_thermal_queries.yaml"

# Check config file
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

# Check dataset split file
SPLIT_FILE="./data/dataset_split.json"
if [ ! -f "$SPLIT_FILE" ]; then
    echo "⚠️  Dataset split file not found: $SPLIT_FILE"
    echo "Please run first: python utils/fixed_split.py --parquet /path/to/data.parquet"
    exit 1
fi

echo ""
echo "Config file: $CONFIG"
echo "Dataset split: $SPLIT_FILE"
echo ""

# Start training
echo "Starting training..."
python scripts/pretrain_thermal_queries.py \
    --config $CONFIG \
    "$@"

echo ""
echo "========================================"
echo "✅ Training Complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to: ./checkpoints/thermal_query_decoder/"
echo "TensorBoard logs: ./checkpoints/thermal_query_decoder/logs/"
echo ""
echo "View training logs:"
echo "  tensorboard --logdir ./checkpoints/thermal_query_decoder/logs/"
echo ""
echo "Next steps:"
echo "1. Verify best model: ls -lh ./checkpoints/thermal_query_decoder/best_model.pth"
echo "2. Proceed to Stage 2: Update ICEdit config with pretrained path"
echo ""
echo "Stage 2 config file:"
echo "  ICEdit_contrastive/train/train/config/thermal_query_joint.yaml"
echo ""
echo "Stage 2 training command:"
echo "  cd ICEdit_contrastive/train"
echo "  export XFL_CONFIG=train/config/thermal_query_joint.yaml"
echo "  python -m src.train.train"
echo "========================================"
