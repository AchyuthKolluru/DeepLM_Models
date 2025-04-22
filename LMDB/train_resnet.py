import os, argparse, lmdb, pickle, io
torch_import = __import__('torch')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
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
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, args.num_classes)
    model = net.to(device)
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