import os, argparse, lmdb, pickle, io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class LMDBDataset(Dataset):
    def __init__(self, path, transform=None):
        if not os.path.exists(path): raise FileNotFoundError(f"LMDB not found: {path}")
        self.env = lmdb.open(path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        self.transform = transform
    def __len__(self): return self.length
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(f"{idx}".encode())
        data = pickle.loads(raw)
        img = Image.open(io.BytesIO(data['image'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, data['label']

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, token_dim, num_classes, depth, heads):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size//patch_size)**2
        self.patch_embed = nn.Conv2d(3, token_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,token_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, token_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.to_cls = nn.Linear(token_dim, num_classes)
    def forward(self,x):
        x = self.patch_embed(x).flatten(2).transpose(1,2)
        b, n, _ = x.size()
        cls = self.cls_token.expand(b,-1,-1)
        x = torch.cat((cls,x), dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.to_cls(x[:,0])

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lmdb_path',required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--num_classes',type=int,default=100)
    p.add_argument('--img_size',type=int,default=224)
    p.add_argument('--token_dim',type=int,default=128)
    p.add_argument('--num_transformer_layers',type=int,default=6)
    p.add_argument('--num_heads',type=int,default=8)
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    tf = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor()])
    ds = LMDBDataset(args.lmdb_path, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = ViT(args.img_size, patch_size=16, token_dim=args.token_dim, num_classes=args.num_classes,
                depth=args.num_transformer_layers, heads=args.num_heads).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x),y).backward()
            opt.step()
        print(f"Epoch {ep+1}/{args.epochs} done")