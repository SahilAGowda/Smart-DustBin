# Smart Dustbin — Waste Classification System

A comprehensive end-to-end system for a smart waste-classifying dustbin that can identify different types of waste materials using computer vision and provide recycling guidance.

The system allows users to upload or capture photos of waste items, uses a CNN model to classify the objects, determines the waste category (dry waste, wet waste, recyclable, hazardous, etc.), and provides detailed information about recyclability and proper disposal methods.

## Project Structure
```text
smart-dustbin/
├── backend/
│   ├── app.py                 # Flask API + static serving
│   ├── models/
│   │   ├── waste_cnn.py       # CNN loader + predict wrapper
│   │   └── label_map.json     # Class index -> label mapping
│   └── utils/
│       └── preprocess.py      # Image loading & transforms
├── frontend/
│   ├── index.html             # Login page
│   ├── dashboard.html         # Upload/scan interface
│   ├── overview.html          # System overview page
│   └── assets/
│       ├── style.css          # Styling
│       └── app.js             # Frontend JavaScript
├── data/                      # Model weights and sample data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Object Detection**: Uses CNN to identify waste objects from images
- **Waste Classification**: Maps detected objects to waste categories:
  - **Dry Waste**: Paper, cardboard, plastic bottles, metal cans
  - **Wet Waste**: Food scraps, organic matter
  - **Recyclable**: Glass, aluminum, certain plastics
  - **Hazardous**: Batteries, electronics, medical waste
  - **E-waste**: Electronic devices, circuit boards
- **Recycling Information**: Provides recycling percentage and disposal guidance
- **User-Friendly Interface**: Simple web interface for easy interaction

## Quick Start

1) **Set up virtual environment and install dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2) **Run the backend server:**
```bash
cd backend
python app.py
```

3) **Access the application:**
Open http://127.0.0.1:5000 in your browser

4) **Using the system:**
   - Login with any username/password (prototype mode)
   - Go to dashboard and upload an image of waste
   - Get classification results with recycling information

## Waste Categories

The system classifies waste into the following categories:

| Category | Examples | Recyclability | Disposal Method |
|----------|----------|---------------|-----------------|
| **Plastic** | Bottles, containers, bags | 40% | Recycling bins, avoid single-use |
| **Metal** | Aluminum cans, steel items | 75% | Metal recycling facilities |
| **Paper** | Newspapers, cardboard | 65% | Paper recycling bins |
| **Glass** | Jars, bottles | 85% | Glass recycling containers |
| **Organic** | Food waste, leaves | 0%* | Composting (*not recyclable but compostable) |
| **E-waste** | Phones, laptops, batteries | 30% | Specialized e-waste facilities |
| **Medical** | Syringes, medical supplies | 5% | Hazardous waste disposal |
| **Other** | Mixed/unidentified items | 10% | General waste assessment needed |

## Integrating a Real CNN Model

To use a real trained model instead of the mock classifier:

1. **Place your model file** at `data/waste_cnn.pt` (PyTorch) or adjust the path in `waste_cnn.py`

2. **Update the model loading code** in `backend/models/waste_cnn.py`:
   - Set `MODEL_AVAILABLE = True`
   - Implement the `load_model()` function for your framework
   - Implement the `predict_tensor()` function with real inference

3. **Example for PyTorch:**
```python
def load_model(self, weights_path: Path):
    import torch
    model = YourCNNModel(num_classes=len(self.label_map))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict_tensor(self, x):
    import torch
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)
        logits = self.model(x_tensor)
        prob = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(prob, dim=1)
        return int(idx.item()), float(confidence.item())
```

## API Endpoints

- `POST /api/login` - User authentication (prototype)
- `POST /api/predict` - Image classification and waste analysis
- `GET /` - Serves the frontend application

## Environmental Impact

This smart dustbin system helps:
- **Reduce contamination** in recycling streams
- **Increase recycling rates** through proper sorting
- **Educate users** about waste categories and disposal
- **Track waste patterns** for better waste management
- **Promote sustainable practices** in communities

## Future Enhancements

- Real-time camera integration
- Database for user analytics
- Multi-language support
- Integration with local waste management systems
- Mobile app development
- Advanced CNN models with higher accuracy
- IoT sensor integration for automatic detection

## Technical Notes

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **AI/ML**: CNN for object detection and classification
- **Image Processing**: PIL/Pillow for image handling
- **CORS enabled** for flexible deployment options
- **Responsive design** for various screen sizes

## Contributing

This is a prototype system designed for educational and demonstration purposes. For production deployment, consider:
- Implementing proper authentication (JWT tokens)
- Adding database integration for user data and analytics
- Implementing proper error handling and logging
- Adding comprehensive testing
- Security hardening for production environments

---

**Note**: This system is designed to promote environmental awareness and proper waste disposal practices. Always follow local waste management guidelines and regulations.
