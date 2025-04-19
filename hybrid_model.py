import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=100, img_size=224, token_dim=128, num_transformer_layers=6, num_heads=8):
        """
        Args:
            num_classes (int): Number of output classes.
            img_size (int): Input image size (assumed square).
            token_dim (int): Dimension of tokens after CNN backbone.
            num_transformer_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in transformer.
        """
        super(HybridCNNTransformer, self).__init__()
        
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )
        
        self.feature_map_size = img_size // 8 
        self.num_tokens = self.feature_map_size * self.feature_map_size

        # Process Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, token_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, dim_feedforward=token_dim * 4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.classifier = nn.Linear(token_dim, num_classes)
        
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        """
        x: input images of shape (batch_size, 3, img_size, img_size)
        """
        batch_size = x.size(0)
        features = self.cnn_backbone(x)
        features = features.flatten(2) 
        features = features.transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, features), dim=1)
        
        tokens = tokens + self.pos_embedding
        
        # Prepare tokens for transformer (PyTorch expects shape: [sequence_length, batch_size, embedding_dim])
        tokens = tokens.transpose(0, 1)
        encoded_tokens = self.transformer_encoder(tokens)
        cls_output = encoded_tokens[0]
        logits = self.classifier(cls_output)
        return logits

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a complex HybridCNNTransformer on a real dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root directory of your dataset (organized in ImageFolder structure).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes in the dataset.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (assumes square images).")
    parser.add_argument("--token_dim", type=int, default=128, help="Dimension of tokens in the model.")
    parser.add_argument("--num_transformer_layers", type=int, default=6, help="Number of transformer encoder layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in transformer encoder.")
    args = parser.parse_args()
    return args

# Main Training Function
def main():
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Initialize model
    model = HybridCNNTransformer(num_classes=args.num_classes, img_size=args.img_size,
                                 token_dim=args.token_dim, num_transformer_layers=args.num_transformer_layers,
                                 num_heads=args.num_heads)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
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
            
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed:.2f} seconds")
    print("Training finished.")

if __name__ == '__main__':
    main()
