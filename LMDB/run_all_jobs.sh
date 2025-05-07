set -e

export PYTHONPATH=/nfs-share/DL/pytorch_lmdb_imagenet:$PYTHONPATH

DATA_DIR=/nfs-share/DL/pytorch_lmdb_imagenet/dataset/tiny-imagenet-lmdb
GPU=0

echo "Running all Models $GPU"

python densenet121_train.py $DATA_DIR --lmdb --gpu $GPU --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python mobilenetv2_train.py $DATA_DIR --lmdb --gpu $GPU --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python efficientnet_b0_train.py $DATA_DIR --lmdb --gpu $GPU --epochs 90 --batch-size 256 --workers 4 --lr 0.1 --pretrained
python vgg16_eval.py $DATA_DIR --gpu $GPU

echo "All benchmarks completed on GPU $GPU."