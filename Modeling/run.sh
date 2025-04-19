LMDB_PATH=""
DEVICE="cuda"
EPOCHS=12
BATCH_SIZE=128
NUM_WORKERS=8
LR=1e-4
NUM_CLASSES=100
IMG_SIZE=224
TOKEN_DIM=128
TRANSFORMER_LAYERS=6
HEADS=8

#Change File depending on Model needed to run
python train_cnn.py \
    --lmdb_path "$LMDB_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --token_dim $TOKEN_DIM \
    --num_transformer_layers $TRANSFORMER_LAYERS \
    --num_heads $HEADS \
    --device $DEVICE