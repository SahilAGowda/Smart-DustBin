"""
Setup script for Smart Dustbin waste classification system.
This script helps you set up the environment and download sample data.

Usage:
python setup.py --install-deps    # Install PyTorch and dependencies
python setup.py --create-dataset  # Create sample dataset structure
python setup.py --download-sample # Download sample model (if available)
python setup.py --all            # Do everything
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        # Install PyTorch (CPU version for compatibility)
        print("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", 
            "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install other requirements
        print("Installing other requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt"
        ])
        
        print("✅ Dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True

def create_dataset_structure():
    """Create the dataset directory structure"""
    print("Creating dataset structure...")
    
    classes = [
        "plastic_bottle", "aluminum_can", "cardboard", "paper", "glass_jar", 
        "food_waste", "old_phone", "laptop", "syringe", "battery", "other"
    ]
    
    # Create directories
    for split in ["train", "val", "test"]:
        for class_name in classes:
            dir_path = Path(f"data/dataset/{split}/{class_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create info file
    dataset_info = {
        "name": "Waste Classification Dataset",
        "classes": classes,
        "num_classes": len(classes),
        "splits": ["train", "val", "test"],
        "description": "Dataset for training waste classification CNN model",
        "instructions": [
            "Add training images to data/dataset/train/{class_name}/",
            "Add validation images to data/dataset/val/{class_name}/", 
            "Add test images to data/dataset/test/{class_name}/",
            "Recommended: 70% train, 20% val, 10% test split",
            "Minimum 50 images per class for decent training"
        ]
    }
    
    with open("data/dataset/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("✅ Dataset structure created in data/dataset/")
    print("📁 Folders created for each waste class")
    print("📄 See data/dataset/dataset_info.json for details")

def create_sample_readme():
    """Create README for the data folder"""
    readme_content = """# Data Folder

This folder contains the datasets and trained models for the Smart Dustbin system.

## Structure

```
data/
├── dataset/                    # Training dataset
│   ├── train/                  # Training images
│   ├── val/                    # Validation images  
│   ├── test/                   # Test images
│   └── dataset_info.json       # Dataset information
├── waste_cnn.pt               # Trained model weights
└── README.md                  # This file
```

## Dataset Classes

The system recognizes these waste categories:

1. **plastic_bottle** - Plastic bottles and containers
2. **aluminum_can** - Aluminum cans and metal containers  
3. **cardboard** - Cardboard boxes and packaging
4. **paper** - Paper documents, newspapers
5. **glass_jar** - Glass bottles and jars
6. **food_waste** - Organic food scraps
7. **old_phone** - Mobile phones and small electronics
8. **laptop** - Laptops and large electronics
9. **syringe** - Medical waste (syringes, etc.)
10. **battery** - Batteries and power cells
11. **other** - Unclassified waste items

## Adding Your Data

1. **Collect Images**: Gather images for each waste category
2. **Organize**: Place images in the corresponding class folders
3. **Split**: Distribute images across train/val/test folders
4. **Train**: Use `python train_model.py` to train the model

## Model Training Tips

- **Minimum 50 images per class** for basic training
- **200+ images per class** for good performance
- **Use diverse lighting and angles** for robustness
- **Balance dataset sizes** across classes
- **Augmentation is applied automatically** during training

## Pre-trained Models

If you don't have a custom dataset, the system will use:
- **ResNet50 backbone** with random classification layers
- **Mock predictions** based on image brightness (demo mode)

For production use, train with your specific waste images!
"""
    
    with open("data/README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created data/README.md with detailed instructions")

def check_system():
    """Check system requirements and setup"""
    print("Checking system setup...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️  Python 3.7+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CUDA not available, using CPU")
    except ImportError:
        print("📦 PyTorch not installed yet")
    
    # Check directory structure
    required_dirs = [
        "backend/models",
        "backend/utils", 
        "frontend/assets",
        "data"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")

def main():
    parser = argparse.ArgumentParser(description="Setup Smart Dustbin system")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install PyTorch and dependencies")
    parser.add_argument("--create-dataset", action="store_true",
                       help="Create dataset directory structure")
    parser.add_argument("--check-system", action="store_true",
                       help="Check system requirements")
    parser.add_argument("--all", action="store_true",
                       help="Run all setup steps")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print("Smart Dustbin Setup Script")
        print("=" * 40)
        print("Available options:")
        print("  --install-deps     Install dependencies")
        print("  --create-dataset   Create dataset structure")
        print("  --check-system     Check system setup")
        print("  --all             Run all steps")
        print("\nExample: python setup.py --all")
        return
    
    print("🗑️  Smart Dustbin Setup")
    print("=" * 40)
    
    if args.check_system or args.all:
        check_system()
        print()
    
    if args.install_deps or args.all:
        install_dependencies()
        print()
    
    if args.create_dataset or args.all:
        create_dataset_structure()
        create_sample_readme()
        print()
    
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your waste images to data/dataset/ folders")
    print("2. Train the model: python train_model.py") 
    print("3. Test the model: python test_model.py --test_folder data/dataset/test")
    print("4. Run the app: cd backend && python app.py")

if __name__ == "__main__":
    main()
