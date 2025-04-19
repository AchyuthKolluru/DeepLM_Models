#!/usr/bin/env python
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# Custom Complex Model
class ComplexCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block adjusts dimensions and downsamples.
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Model Factory Function
def get_model(model_name, num_classes, pretrained=False):
    """
    Returns the model instance based on the provided model_name.
    Supported models: 'complex', 'resnet50', 'vgg16'.
    """
    model_name = model_name.lower()
    if model_name == "complex":
        model = ComplexCNN(num_classes=num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Unsupported model. Choose from 'complex', 'resnet50', or 'vgg16'.")
    return model


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on the Oxford 102 Flowers dataset using various architectures.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root directory where the Flowers102 dataset will be downloaded/stored.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_classes", type=int, default=102, help="Number of classes (102 for Flowers102).")
    parser.add_argument("--model", type=str, default="resnet50", choices=["complex", "resnet50", "vgg16"],
                        help="Model architecture to use: 'complex', 'resnet50', or 'vgg16'.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained weights (applicable for resnet50 and vgg16).")
    args = parser.parse_args()
    return args


# Main Training Loop
def main():
    args = parse_args()

    # Device setup: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.Flowers102(root=args.data_dir, split='train', transform=transform_train, download=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed:.2f} seconds")
    print("Training finished.")

if __name__ == '__main__':
    main()
