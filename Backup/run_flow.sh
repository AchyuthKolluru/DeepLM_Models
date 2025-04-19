DATA_DIR=""

OUTPUT_DIR=""

python flow_estimation.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name raft_large \
    --pretrained \
    --batch_size 2 \
    --num_workers 8 \
    --img_size 512 \
    --iterations 20
