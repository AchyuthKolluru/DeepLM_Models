import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, token_dim, num_classes, depth, heads):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, token_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_cls = nn.Linear(token_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.to_cls(x[:, 0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--token_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
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
    model = ViT(args.img_size, args.patch_size, args.token_dim, args.num_classes, args.depth, args.heads).to(device)
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