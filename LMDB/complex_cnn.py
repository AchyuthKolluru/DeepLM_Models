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

from folder2lmdb import ImageFolderLMDB

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

best_acc1 = 0

class ComplexCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    parser = argparse.ArgumentParser(description='Train ComplexCNN on Tiny-ImageNet')
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456')
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--lmdb', action='store_true')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Seeding enabled – this may slow down training.")

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus * args.world_size
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), device=(device))
    else:
        main_worker(args.gpu, ngpus, args, device)

def main_worker(gpu, ngpus_per_node, args, device):
    global best_acc1
    args.gpu = gpu

    print("=> creating ComplexCNN model")
    model = ComplexCNN()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()
    


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume and os.path.isfile(args.resume):
        loc = f'cuda:{gpu}' if gpu is not None else None
        ckpt = torch.load(args.resume, map_location=loc)
        args.start_epoch = ckpt['epoch']
        best_acc1 = ckpt['best_acc1']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"=> resumed from {args.resume} (epoch {args.start_epoch})")

    cudnn.benchmark = True

    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if not args.lmdb:
        raise RuntimeError("This script only supports LMDB (`--lmdb`)")
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

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                      shuffle=(train_sampler is None), num_workers=args.workers,
                      pin_memory=True, sampler=train_sampler)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                      shuffle=False, num_workers=args.workers, pin_memory=True)

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
        if not args.multiprocessing_distributed or (args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

def train(loader, model, criterion, optimizer, epoch, args):
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(loader):
        data_time = time.time() - end
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            batch_time = time.time() - end
            print(f"Epoch [{epoch}][{i+1}/{len(loader)}]\t"
                  f"Time {batch_time:.3f} ({data_time:.3f})\t"
                  f"Loss {loss.item():.4f}\t"
                  f"Acc@1 {acc1[0]:.3f}\t"
                  f"Acc@5 {acc5[0]:.3f}")
        end = time.time()

def validate(loader, model, criterion, args):
    model.eval()
    losses, top1 = 0.0, 0.0
    with torch.no_grad():
        for images, target in loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1,5))
            losses += loss.item()
            top1 += acc1[0].item()

    avg_acc1 = top1 / len(loader)
    print(f" * Acc@1 {avg_acc1:.3f}")
    return avg_acc1

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for g in optimizer.param_groups:
        g['lr'] = lr

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk); batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            c_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(c_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()