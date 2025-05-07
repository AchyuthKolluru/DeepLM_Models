import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from folder2lmdb import ImageFolderLMDB
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description='Quick VGG16 Val on Tiny-ImageNet')
    parser.add_argument('data', help='path to lmdb dataset')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    gpu = args.gpu

    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 200)
    model = model.cuda(gpu) if gpu is not None else nn.DataParallel(model).cuda()

    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    ds = ImageFolderLMDB(os.path.join(args.data, 'val.lmdb'), tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, tgt in loader:
            if gpu is not None:
                imgs, tgt = imgs.cuda(gpu), tgt.cuda(gpu)
            out = model(imgs)
            _, p = out.topk(1, 1, True, True)
            correct += (p.view(-1) == tgt).sum().item()
            total += tgt.size(0)
    print(f"VGG16 Val Acc: {correct/total*100:.2f}%")

if __name__ == '__main__':
    main()