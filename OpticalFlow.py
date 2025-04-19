import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------
# Dataset: Sequential Frame Pairs
# ----------------------------
class FlowDataset(Dataset):
    def __init__(self, root_dir, img_size=None, transform=None):
        """
        root_dir should contain subfolders (one per video/sequence),
        each with ordered frame images (e.g., frame0001.png, frame0002.png, ...).
        """
        self.root = root_dir
        self.transform = transform
        self.pairs = []
        # Build list of (frame_t, frame_{t+1}) paths
        for seq in sorted(os.listdir(root_dir)):
            seq_dir = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            frames = sorted(
                f for f in os.listdir(seq_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
            for i in range(len(frames) - 1):
                self.pairs.append((
                    os.path.join(seq_dir, frames[i]),
                    os.path.join(seq_dir, frames[i + 1])
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2 = self.pairs[idx]
        img1 = Image.open(f1).convert('RGB')
        img2 = Image.open(f2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, f1, f2

# ----------------------------
# Argument Parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate optical flow on a video dataset with RAFT."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of your video dataset (ImageFolder style).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the .npy flow files.")
    parser.add_argument("--model_name", type=str, default="raft_large",
                        choices=["raft_large", "raft_small"],
                        help="Which RAFT variant to use.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained weights (default: True for RAFT).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of frame‑pairs per forward).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers.")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Resize shorter side of frames to this size (keep aspect ratio).")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of RAFT iterations (more = slower, more accurate).")
    args = parser.parse_args()
    return args

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load RAFT from torch.hub
    print(f"[INFO] Loading {args.model_name}...")
    model = torch.hub.load("princeton-vl/RAFT", args.model_name, pretrained=args.pretrained)
    model = model.module if hasattr(model, "module") else model
    model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Found {torch.cuda.device_count()} GPUs, using DataParallel")
        model = nn.DataParallel(model)

    transform_list = []
    if args.img_size:
        transform_list.append(transforms.Resize(args.img_size))
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]
    transform = transforms.Compose(transform_list)

    # Dataset & DataLoader
    dataset = FlowDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Inference loop
    total_pairs = len(dataset)
    print(f"[INFO] {total_pairs} frame‑pairs found. Beginning inference...")
    start_all = time.time()
    for idx, (img1, img2, path1, path2) in enumerate(loader):
        t0 = time.time()
        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            flow_low, flow_up = model(img1, img2,
                                      iters=args.iterations,
                                      test_mode=True)
        # flow_up: [B,2,H,W]
        flows = flow_up.cpu().numpy()  # (B,2,H,W)

        # Save each in batch
        for b in range(flows.shape[0]):
            base1 = os.path.splitext(os.path.basename(path1[b]))[0]
            base2 = os.path.splitext(os.path.basename(path2[b]))[0]
            save_name = f"{base1}__{base2}.npy"
            save_path = os.path.join(args.output_dir, save_name)
            # transpose to (H,W,2) for easy loading
            np.save(save_path, flows[b].transpose(1,2,0))

        t1 = time.time()
        print(f"[{idx+1}/{total_pairs}] Batch time: {t1-t0:.2f}s, saved {flows.shape[0]} flows")

    elapsed = time.time() - start_all
    print(f"[DONE] Processed {total_pairs} pairs in {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
