#!/usr/bin/env bash

# Directory containing train.lmdb / val.lmdb
DATA_DIR=/nfs-share/DL/pytorch_lmdb_imagenet/dataset/tiny-imagenet-lmdb

WORKERS=4
EPOCHS=90
BATCH_SIZE=128
LR=0.1
GPU=0

echo "=== Training ComplexCNN ==="
python complex_cnn.py \
    $DATA_DIR \
    --lmdb \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr $LR    \
    --gpu $GPU

echo "=== Training ResNet50 ==="
python resnet50_train.py \
    $DATA_DIR \
    --lmdb \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr $LR \
    --pretrained
    --gpu $GPU

echo "=== Training ViT (B-16) ==="
python vit_train.py \
    $DATA_DIR \
    --lmdb \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr $LR \
    --pretrained    \
    --gpu $GPU
