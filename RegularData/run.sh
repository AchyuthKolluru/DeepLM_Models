set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cnn|resnet|vit>"
  exit 1
fi

MODEL="$1"
shift

# Configuration
DATA_DIR="/nfs-share/datasets"
DEVICE="cuda"
EPOCHS=12
BATCH_SIZE=128
NUM_WORKERS=8
LR=1e-4
NUM_CLASSES=100
IMG_SIZE=224

# Modified so that the commands for ViT are separated

# ViT-specific (only used if MODEL=vit)
PATCH_SIZE=16
TOKEN_DIM=128
NUM_LAYERS=6
HEADS=8

# Select the correct training script
case "$MODEL" in
  cnn)
    SCRIPT="train_cnn.py"
    EXTRA_ARGS=""
    ;;
  resnet)
    SCRIPT="train_resnet.py"
    EXTRA_ARGS=""
    ;;
  vit)
    SCRIPT="train_vit.py"
    EXTRA_ARGS="--patch_size $PATCH_SIZE \
                --token_dim $TOKEN_DIM \
                --num_transformer_layers $NUM_LAYERS \
                --num_heads $HEADS"
    ;;
  *)
    echo "Invalid model: $MODEL. Choose from cnn, resnet, vit."
    exit 1
    ;;
esac

# Run the selected script
echo ">>> Training $MODEL"
python $SCRIPT \
    --data_dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    $EXTRA_ARGS \
    --device $DEVICE

echo "=== All done ==="