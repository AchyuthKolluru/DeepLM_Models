# resnet50_train.py
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from folder2lmdb import ImageFolderLMDB
from progress.bar import Bar

best_acc1 = 0

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-50 on Tiny-ImageNet')
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('--arch', default='resnet50', choices=['resnet50'])
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
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456')
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--lmdb', action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    print(f"=> creating model '{args.arch}'")
    model = models.resnet50(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, 200)  # tiny-imagenet has 200 classes

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume and os.path.isfile(args.resume):
        loc = f'cuda:{args.gpu}' if args.gpu is not None else None
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> resumed from {args.resume} (epoch {args.start_epoch})")

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
    if args.lmdb:
        traindir = os.path.join(args.data, 'train.lmdb')
        valdir = os.path.join(args.data, 'val.lmdb')
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
    else:
        raise RuntimeError("Only LMDB supported (use `--lmdb`)")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) \
                    if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(getattr(args, 'start_epoch', 0), args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if not args.multiprocessing_distributed or \
           (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

# (train, validate, save_checkpoint, adjust_learning_rate, accuracy same as in complex_cnn.py)
# For brevity, copy those implementations verbatim here.
from complex_cnn import train, validate, save_checkpoint, adjust_learning_rate, accuracy

if __name__ == '__main__':
    main()
