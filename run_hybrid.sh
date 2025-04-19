#!/bin/bash

# source /path/to/your/env/bin/activate

DATA_DIR=' '

# Run the script
python train_complex_model.py --data_dir "$DATA_DIR" \
    --epochs 10 \
    --batch_size 64 \
    --num_workers 8 \
    --lr 0.0001 \
    --num_classes 100 \
    --img_size 224 \
    --token_dim 128 \
    --num_transformer_layers 6 \
    --num_heads 8
