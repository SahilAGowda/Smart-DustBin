# 🧠 CNN Model Integration Guide

This guide explains how to use the integrated CNN model for waste classification in your Smart Dustbin system.

## 🚀 Quick Start

### Option 1: Use Pre-trained ResNet50 (Recommended for Testing)

The system is now configured to use a ResNet50-based CNN model. Even without custom training, it will work with transfer learning:

```bash
# 1. Install dependencies
python setup.py --install-deps

# 2. Run the system
cd backend
python app.py
```

The model will automatically use a pre-trained ResNet50 with custom classification layers.

### Option 2: Train Your Own Model

For best results, train with your own waste images:

```bash
# 1. Create dataset structure
python setup.py --create-dataset

# 2. Add your images to data/dataset/train/ and data/dataset/val/
# Each class should have its own folder (plastic_bottle, aluminum_can, etc.)

# 3. Train the model
python train_model.py --epochs 50 --batch_size 32

# 4. Test the trained model
python test_model.py --test_folder data/dataset/test/

# 5. Run the web application
cd backend
python app.py
```

## 📊 Model Architecture

The CNN model uses:
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Classification Head**: 
  - Dropout(0.5)
  - Linear(2048 → 512)
  - ReLU activation
  - Dropout(0.3)
  - Linear(512 → num_classes)

## 🗂️ Dataset Structure

Organize your images like this:

```
data/
└── dataset/
    ├── train/
    │   ├── plastic_bottle/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   ├── aluminum_can/
    │   ├── cardboard/
    │   ├── paper/
    │   ├── glass_jar/
    │   ├── food_waste/
    │   ├── old_phone/
    │   ├── laptop/
    │   ├── syringe/
    │   ├── battery/
    │   └── other/
    └── val/
        └── (same structure as train)
```

## 🏋️ Training Tips

### Data Collection
- **50+ images per class** minimum
- **200+ images per class** for good performance
- Use **diverse lighting conditions**
- Include **different angles and backgrounds**
- **Balance class sizes** (similar number of images per class)

### Training Parameters
```bash
# Basic training
python train_model.py --epochs 30 --batch_size 16

# Advanced training (if you have lots of data and GPU)
python train_model.py --epochs 100 --batch_size 64 --learning_rate 0.0001
```

### Data Augmentation (Applied Automatically)
- Random resizing and cropping
- Horizontal flipping
- Rotation (±10 degrees)
- Color jittering (brightness, contrast, saturation)

## 🔧 Model Configuration

### Key Files
- `backend/models/waste_cnn.py` - Model definition and loading
- `backend/models/label_map.json` - Class index to label mapping
- `data/waste_cnn.pt` - Trained model weights (created after training)

### Model Status
The web interface shows:
- **CNN Model**: Real trained model is loaded
- **Mock Model**: No trained model, using fallback predictions
- **Device**: CPU or CUDA GPU being used

## 🧪 Testing Your Model

### Test Single Image
```bash
python test_model.py --image_path path/to/image.jpg
```

### Test Multiple Images
```bash
python test_model.py --test_folder path/to/test/images/
```

### Example Output
```
Testing image: test_images/bottle.jpg
Image size: (640, 480)
Predicted class: plastic_bottle
Confidence: 0.892
Waste category: plastic
```

## 🎯 Waste Classification Logic

The system maps detected objects to waste categories:

| Detected Object | Waste Category | Recyclability |
|----------------|----------------|---------------|
| plastic_bottle | plastic | 40% |
| aluminum_can | metal | 75% |
| cardboard, paper | paper | 65% |
| glass_jar | glass | 85% |
| food_waste | organic | 0% (compostable) |
| old_phone, laptop | e-waste | 30% |
| syringe | medical | 5% |
| battery | e-waste | 30% |
| other | other | 10% |

## 🚨 Troubleshooting

### Model Not Loading
```
Model status: mock
```
**Solution**: Train a model or check if `data/waste_cnn.pt` exists

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: The model automatically falls back to CPU

### Low Accuracy
```
Validation accuracy: 45%
```
**Solutions**:
- Add more training data
- Train for more epochs
- Check data quality and labeling
- Ensure balanced dataset

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: 
```bash
python setup.py --install-deps
```

## 📈 Performance Expectations

### With Good Dataset (200+ images per class)
- **Training Time**: 30-60 minutes (CPU), 5-15 minutes (GPU)
- **Expected Accuracy**: 80-95%
- **Confidence Scores**: 0.7-0.95 for correct predictions

### With Minimal Dataset (50 images per class)
- **Training Time**: 10-20 minutes (CPU), 2-5 minutes (GPU)
- **Expected Accuracy**: 60-80%
- **Confidence Scores**: 0.5-0.8 for correct predictions

## 🔮 Advanced Features

### Custom Model Architecture
Modify `WasteClassificationCNN` in `waste_cnn.py` to use different backbones:
- ResNet18/34/101/152
- EfficientNet
- MobileNet (for edge deployment)

### Multi-GPU Training
```bash
# If you have multiple GPUs
python train_model.py --batch_size 128 --epochs 50
```

### Model Export for Production
```python
# Convert to TorchScript for faster inference
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("data/waste_cnn_traced.pt")
```

## 📱 Integration with Mobile Apps

The trained model can be exported for mobile deployment:
- **iOS**: Convert to Core ML format
- **Android**: Convert to TensorFlow Lite
- **React Native**: Use ONNX format

---

**Ready to classify waste with AI! 🤖♻️**

For questions or issues, check the troubleshooting section or create an issue in the project repository.
