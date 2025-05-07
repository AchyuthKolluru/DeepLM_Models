import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import random
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from folder2lmdb import ImageFolderLMDB
from complex_cnn import train, validate, save_checkpoint, adjust_learning_rate, accuracy

best_acc1 = 0

def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 on Tiny-ImageNet')
    parser.add_argument('data', help='path to lmdb dataset')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--lmdb', action='store_true')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
    else:
        main_worker(args.gpu, ngpus, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    print("=> creating MobileNetV2 model")
    model = models.mobilenet_v2(pretrained=args.pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 200)
    model = model.cuda(gpu) if gpu is not None else nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    traindir = os.path.join(args.data, 'train.lmdb')
    valdir   = os.path.join(args.data, 'val.lmdb')
    train_dataset = ImageFolderLMDB(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    val_dataset = ImageFolderLMDB(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(getattr(args, 'start_epoch', 0), args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        if acc1 > best_acc1:
            best_acc1 = acc1
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, True)

if __name__ == '__main__':
    main()