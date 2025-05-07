set -e

export CUDA_VISIBLE_DEVICES=""

DATA_DIR=/nfs-share/DL/pytorch_lmdb_imagenet/dataset/tiny-imagenet-lmdb

echo "Running all benchmarks on CPU only"

python densenet121_train.py   $DATA_DIR --lmdb --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python mobilenetv2_train.py  $DATA_DIR --lmdb --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python efficientnet_b0_train.py $DATA_DIR --lmdb --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python vgg16_eval.py         $DATA_DIR

echo "All CPU-only benchmarks completed."