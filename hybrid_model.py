import argparse
import io
import time

import lmdb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------
# Dataset: LMDB-backed
# ----------------------------
class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, img_size, transform=None):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            length_bytes = txn.get(b'__len__')
            if length_bytes is None:
                raise KeyError("`__len__` key not found in LMDB")
            self.length = int(length_bytes.decode('ascii'))

        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            img_key = f"image-{idx:08d}".encode('ascii')
            lbl_key = f"label-{idx:08d}".encode('ascii')

            img_bytes = txn.get(img_key)
            if img_bytes is None:
                raise KeyError(f"Image bytes for key {img_key} not found")
            label_bytes = txn.get(lbl_key)
            if label_bytes is None:
                raise KeyError(f"Label bytes for key {lbl_key} not found")

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label_bytes.decode('ascii'))
        return img, label

# ----------------------------
# HybridCNNTransformer Model
# ----------------------------
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=100, img_size=224, token_dim=128,
                 num_transformer_layers=6, num_heads=8):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(token_dim), nn.ReLU(inplace=True)
        )
        self.feature_map_size = img_size // 8
        self.num_tokens = self.feature_map_size * self.feature_map_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.classifier = nn.Linear(token_dim, num_classes)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        b = x.size(0)
        feat = self.cnn_backbone(x)
        feat = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embedding
        tokens = tokens.transpose(0, 1)
        encoded = self.transformer_encoder(tokens)
        cls_out = encoded[0]
        return self.classifier(cls_out)

# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train HybridCNNTransformer on an LMDB dataset"
    )
    p.add_argument("--lmdb_path", type=str, required=True,
                   help="Path to train.lmdb directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--token_dim", type=int, default=128)
    p.add_argument("--num_transformer_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device to run on: 'cuda' or 'cpu'.")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = LmdbDataset(args.lmdb_path, args.img_size, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = HybridCNNTransformer(
        num_classes=args.num_classes,
        img_size=args.img_size,
        token_dim=args.token_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads
    )
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INFO] {torch.cuda.device_count()} GPUs detected, using DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for i, (imgs, labels) in enumerate(loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch:02d}] Batch {i:04d}  Loss: {epoch_loss / i:.4f}")

        print(f"[Epoch {epoch:02d}] Avg Loss: {epoch_loss / len(loader):.4f}  Time: {time.time() - t0:.1f}s")

    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()