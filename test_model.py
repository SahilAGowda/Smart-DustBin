"""
Test script for the waste classification model.
This script allows you to test the trained model with individual images.

Usage:
python test_model.py --image_path path/to/image.jpg
python test_model.py --test_folder path/to/test/images/
"""

import torch
from PIL import Image
import argparse
from pathlib import Path
import json
import sys

# Add backend to path
sys.path.append('backend')
from models.waste_cnn import WasteCNN

def test_single_image(model, image_path):
    """Test model on a single image"""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        print(f"\nTesting image: {image_path}")
        print(f"Image size: {image.size}")
        
        # Get prediction
        idx, confidence = model.predict_image(image)
        label = model.idx_to_label(idx)
        
        print(f"Predicted class: {label}")
        print(f"Confidence: {confidence:.3f}")
        
        # Map to waste type
        waste_mapping = {
            "plastic_bottle": "plastic",
            "aluminum_can": "metal",
            "cardboard": "paper", 
            "paper": "paper",
            "glass_jar": "glass",
            "food_waste": "organic",
            "old_phone": "e-waste",
            "laptop": "e-waste",
            "syringe": "medical",
            "battery": "e-waste",
            "other": "other"
        }
        
        waste_type = waste_mapping.get(label, "other")
        print(f"Waste category: {waste_type}")
        
        return label, confidence, waste_type
        
    except Exception as e:
        print(f"Error testing image {image_path}: {e}")
        return None, 0.0, "error"

def test_folder(model, folder_path):
    """Test model on all images in a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist!")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in {folder_path}")
    print("=" * 50)
    
    results = []
    for image_file in image_files:
        label, confidence, waste_type = test_single_image(model, image_file)
        results.append({
            'file': image_file.name,
            'label': label,
            'confidence': confidence,
            'waste_type': waste_type
        })
        print("-" * 30)
    
    # Summary
    print("\nSUMMARY:")
    print("=" * 50)
    for result in results:
        print(f"{result['file']:20} -> {result['label']:15} ({result['confidence']:.3f}) [{result['waste_type']}]")

def main():
    parser = argparse.ArgumentParser(description="Test waste classification model")
    parser.add_argument("--image_path", type=str, help="Path to single image to test")
    parser.add_argument("--test_folder", type=str, help="Path to folder containing test images")
    parser.add_argument("--model_path", type=str, default="data/waste_cnn.pt", 
                       help="Path to trained model")
    parser.add_argument("--label_map", type=str, default="backend/models/label_map.json",
                       help="Path to label mapping file")
    
    args = parser.parse_args()
    
    if not args.image_path and not args.test_folder:
        print("Please provide either --image_path or --test_folder")
        return
    
    # Initialize model
    print("Loading model...")
    model = WasteCNN(
        label_map_path=args.label_map,
        weights_path=args.model_path
    )
    
    model_info = model.get_model_info()
    print(f"Model status: {model_info['status']}")
    print(f"Device: {model_info['device']}")
    
    if model_info['status'] == 'mock':
        print("\nWARNING: No trained model found, using mock predictions!")
        print("To use a real model, train one using train_model.py")
    
    # Test single image or folder
    if args.image_path:
        if not Path(args.image_path).exists():
            print(f"Image file {args.image_path} does not exist!")
            return
        test_single_image(model, args.image_path)
    
    if args.test_folder:
        test_folder(model, args.test_folder)

if __name__ == "__main__":
    main()
