"""
Training script for waste classification CNN model.

This script demonstrates how to train a CNN model for waste classification.
You'll need to prepare your dataset according to the folder structure shown below.

Dataset Structure:
data/
├── dataset/
│   ├── train/
│   │   ├── plastic_bottle/
│   │   ├── aluminum_can/
│   │   ├── cardboard/
│   │   ├── paper/
│   │   ├── glass_jar/
│   │   ├── food_waste/
│   │   ├── old_phone/
│   │   ├── laptop/
│   │   ├── syringe/
│   │   ├── battery/
│   │   └── other/
│   └── val/
│       ├── plastic_bottle/
│       ├── aluminum_can/
│       └── ... (same structure as train)
└── waste_cnn.pt  # Output model file

Usage:
python train_model.py --data_dir data/dataset --epochs 50 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
from pathlib import Path
import json
import time

# Import our model
import sys
sys.path.append('backend/models')
from waste_cnn import WasteClassificationCNN

def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001, save_path="data/waste_cnn.pt"):
    """Train the waste classification model"""
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"{data_dir}/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = WasteClassificationCNN(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': epoch_loss,
                'class_to_idx': train_dataset.class_to_idx
            }
            torch.save(checkpoint, save_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    
    # Save class mapping for inference
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open("backend/models/label_map.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print("Updated label_map.json with trained classes")

def create_sample_dataset():
    """Create a sample dataset structure for demonstration"""
    print("Creating sample dataset structure...")
    
    classes = ["plastic_bottle", "aluminum_can", "cardboard", "paper", "glass_jar", 
               "food_waste", "old_phone", "laptop", "syringe", "battery", "other"]
    
    for split in ["train", "val"]:
        for class_name in classes:
            Path(f"data/dataset/{split}/{class_name}").mkdir(parents=True, exist_ok=True)
    
    print("Sample dataset structure created in data/dataset/")
    print("Please add your images to the respective folders and run training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train waste classification model")
    parser.add_argument("--data_dir", type=str, default="data/dataset", 
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--save_path", type=str, default="data/waste_cnn.pt", 
                       help="Path to save trained model")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample dataset structure")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
    else:
        if not Path(args.data_dir).exists():
            print(f"Dataset directory {args.data_dir} not found!")
            print("Use --create_sample to create the directory structure first.")
            exit(1)
        
        train_model(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path
        )
