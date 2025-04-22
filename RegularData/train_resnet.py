import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset_root = os.path.dirname(args.data_dir)
    train_set = datasets.CIFAR100(root=dataset_root, train=True, transform=transform, download=False)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, args.num_classes)
    model = net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_fn(model(imgs), labels).backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} complete")