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

best_acc1 = 0

def main():
    parser = argparse.ArgumentParser("Train DenseNet-121 on Tiny-ImageNet")
    parser.add_argument('data')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
    else:
        main_worker(args.gpu, ngpus, args)


def main_worker(gpu, ngpus, args):
    global best_acc1
    if gpu is not None:
        torch.cuda.set_device(gpu)
    model = models.densenet121(pretrained=args.pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, 200)
    model = model.cuda(gpu) if gpu is not None else nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, gpu)
        acc1 = validate(val_loader, model, criterion, gpu)
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), 'densenet121_best.pth')


def train_one_epoch(loader, model, criterion, optimizer, epoch, gpu):
    model.train()
    for i, (images, target) in enumerate(loader):
        if gpu is not None:
            images, target = images.cuda(gpu), target.cuda(gpu)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] Step [{i+1}/{len(loader)}] Loss: {loss.item():.4f}")


def validate(loader, model, criterion, gpu):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in loader:
            if gpu is not None:
                images, target = images.cuda(gpu), target.cuda(gpu)
            output = model(images)
            _, pred = output.topk(1, 1, True, True)
            correct += (pred.view(-1) == target).sum().item()
            total += target.size(0)
    acc = correct / total * 100
    print(f"DenseNet-121 Val Acc: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    main()