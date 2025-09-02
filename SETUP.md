# Smart Dustbin - Setup Guide

## 🚀 Quick Setup Instructions

### Step 1: Install Python Dependencies
```bash
# Navigate to project directory
cd "c:\Users\Sahil\Desktop\smart dustbin"

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
# Navigate to backend directory
cd backend

# Start the Flask server
python app.py
```

### Step 3: Access the Application
Open your web browser and go to: `http://127.0.0.1:5000`

## 🧠 How the CNN Classification Works

The system uses a Convolutional Neural Network (CNN) to classify waste items:

1. **Image Input**: User uploads/captures an image of waste
2. **Preprocessing**: Image is converted to tensor format for CNN processing
3. **Object Detection**: CNN identifies the specific object (e.g., "plastic_bottle", "aluminum_can")
4. **Waste Classification**: Object is mapped to waste category:
   - `plastic_bottle` → `plastic` (40% recyclable)
   - `aluminum_can` → `metal` (75% recyclable)
   - `food_waste` → `organic` (0% recyclable, but compostable)
   - `old_phone` → `e-waste` (30% recyclable)
   - etc.

## 🔧 Integrating Your Own CNN Model

Currently, the system uses mock predictions. To use a real CNN model:

### For PyTorch Models:
1. Place your trained model file at `data/waste_cnn.pt`
2. Edit `backend/models/waste_cnn.py`:
   - Set `MODEL_AVAILABLE = True`
   - Uncomment PyTorch imports
   - Implement the model loading and prediction functions

### Example PyTorch Integration:
```python
# In waste_cnn.py
MODEL_AVAILABLE = True
import torch
import torchvision.transforms as transforms

def load_model(self, weights_path: Path):
    # Define your model architecture
    model = YourCNNModel(num_classes=11)  # 11 classes in label_map.json
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict_tensor(self, x):
    if self.model is not None:
        # Convert numpy array to PyTorch tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        x_tensor = transform(x).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return int(predicted.item()), float(confidence.item())
```

## 📊 Waste Classification Categories

| Object Type | Waste Category | Recyclability | Disposal Method |
|-------------|----------------|---------------|-----------------|
| plastic_bottle | plastic | 40% | Recycling bin |
| aluminum_can | metal | 75% | Metal recycling |
| cardboard, paper | paper | 65% | Paper recycling |
| glass_jar | glass | 85% | Glass recycling |
| food_waste | organic | 0% | Composting |
| old_phone, laptop | e-waste | 30% | E-waste facility |
| syringe | medical | 5% | Hazardous waste |
| battery | e-waste | 30% | Battery recycling |

## 🛠 Training Your Own Model

To train a custom waste classification model:

1. **Collect Dataset**: Gather images of different waste items
2. **Label Data**: Create labels matching the categories in `label_map.json`
3. **Train CNN**: Use frameworks like PyTorch or TensorFlow
4. **Save Model**: Save trained weights to `data/waste_cnn.pt`
5. **Integrate**: Update the model loading code as shown above

## 🌟 Features

- **Web Interface**: Clean, responsive design
- **Image Upload**: Support for various image formats
- **Real-time Classification**: Instant results after upload
- **Recyclability Info**: Shows percentage and disposal guidance
- **Extensible**: Easy to add new waste categories
- **Mock Mode**: Works out of the box for demonstration

## 🔮 Future Enhancements

- Real-time camera capture
- Mobile app version
- Database integration for analytics
- Multi-language support
- Integration with local waste management
- Advanced CNN models with higher accuracy
- IoT sensors for automatic detection

## 📝 Notes

- Currently in prototype mode with mock authentication
- Recyclability percentages are heuristic-based
- For production, implement proper security and error handling
- Consider local waste management regulations

---

**Ready to start classifying waste intelligently! 🗑️♻️**
