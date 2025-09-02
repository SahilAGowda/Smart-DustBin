import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import numpy as np
from PIL import Image

# Toggle this flag to True once your real model is wired in.
MODEL_AVAILABLE = True

class WasteClassificationCNN(nn.Module):
    """Custom CNN model for waste classification using ResNet50 backbone"""
    
    def __init__(self, num_classes=11, pretrained=True):
        super(WasteClassificationCNN, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class WasteCNN:
    def __init__(self, label_map_path: str, weights_path: str = "data/waste_cnn.pt"):
        self.label_map = self._load_label_map(label_map_path)
        self.weights_path = Path(weights_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Using device: {self.device}")
        
        if MODEL_AVAILABLE:
            self.model = self.load_model(self.weights_path)
        else:
            print("Model not available, using mock predictions")

    def _load_label_map(self, p: str):
        with open(p, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        # convert keys to int for easier indexing
        return {int(k): v for k, v in label_map.items()}

    def load_model(self, weights_path: Path):
        """Load real model (PyTorch) when MODEL_AVAILABLE=True."""
        try:
            # Initialize model
            model = WasteClassificationCNN(num_classes=len(self.label_map))
            
            if weights_path.exists():
                print(f"Loading model weights from {weights_path}")
                # Load trained weights
                checkpoint = torch.load(weights_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                
                print("Model weights loaded successfully")
            else:
                print(f"No model weights found at {weights_path}")
                print("Using pre-trained ResNet50 with random final layers")
                # If no trained weights, use pre-trained ResNet50 for demo
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock predictions")
            return None

    def predict_tensor(self, x):
        """Return (class_index, confidence) for a preprocessed input."""
        if self.model is None:
            # ---- Mock path: simple heuristic demo ----
            print("Using mock prediction")
            import numpy as np
            mean_val = float(np.mean(x))
            if mean_val > 0.75:
                idx = 4  # pretend "glass_jar"
            elif mean_val > 0.55:
                idx = 0  # pretend "plastic_bottle"
            elif mean_val > 0.35:
                idx = 1  # pretend "aluminum_can"
            else:
                idx = 10 # "other"
            confidence = 0.42  # mocked
            return idx, confidence
        else:
            # ---- Real CNN prediction ----
            try:
                # Convert numpy array to PIL Image if needed
                if isinstance(x, np.ndarray):
                    # Convert from [0,1] to [0,255] and to uint8
                    x_img = (x * 255).astype(np.uint8)
                    x_pil = Image.fromarray(x_img)
                else:
                    x_pil = x
                
                # Apply transforms
                x_tensor = self.transform(x_pil).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                return int(predicted.item()), float(confidence.item())
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                # Fallback to mock prediction
                return 10, 0.1  # "other" with low confidence

    def predict_image(self, pil_image):
        """Predict directly from PIL Image"""
        return self.predict_tensor(pil_image)

    def idx_to_label(self, idx: int) -> str:
        return self.label_map.get(idx, "other")
    
    def get_model_info(self):
        """Return information about the loaded model"""
        if self.model is None:
            return {"status": "mock", "device": str(self.device)}
        else:
            return {
                "status": "loaded", 
                "device": str(self.device),
                "num_classes": len(self.label_map),
                "model_type": "ResNet50-based CNN"
            }
