set -e

# --- Configuration ---
DATA_DIR="/nfs-share/datasets/cifar-100-python"
DEVICE="cuda"
EPOCHS=12
BATCH_SIZE=128
NUM_WORKERS=8
LR=1e-4
NUM_CLASSES=100
IMG_SIZE=224

# Had to modify so that these specific commands are only for ViT

# ViT‐specific
PATCH_SIZE=16
TOKEN_DIM=128
NUM_LAYERS=6
HEADS=8

# --- Run CNN ---
echo ">>> Training CNN"
python train_cnn.py \
    --data_dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --device $DEVICE

# --- Run ResNet ---
echo ">>> Training ResNet"
python train_resnet.py \
    --data_dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --device $DEVICE

# --- Run Vision Transformer ---
echo ">>> Training ViT"
python train_vit.py \
    --data_dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --patch_size $PATCH_SIZE \
    --token_dim $TOKEN_DIM \
    --num_transformer_layers $NUM_LAYERS \
    --num_heads $HEADS \
    --device $DEVICE

echo "=== All done ==="
