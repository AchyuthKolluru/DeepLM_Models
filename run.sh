DATA_DIR=' '

# Train using the custom ComplexCNN model
python train.py --data_dir "$DATA_DIR" --epochs 10 --batch_size 128 --num_workers 8 --lr 0.0001 --num_classes 102 --model complex

# Train using ResNet50 with pretrained weights.
# python train.py --data_dir "$DATA_DIR" --epochs 10 --batch_size 128 --num_workers 8 --lr 0.0001 --num_classes 102 --model resnet50 --pretrained

 # Train using VGG16 with pretrained weights.
# python train.py --data_dir "$DATA_DIR" --epochs 10 --batch_size 128 --num_workers 8 --lr 0.0001 --num_classes 102 --model vgg16 --pretrained