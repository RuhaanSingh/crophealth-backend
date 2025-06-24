import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class MLPredictionService:
    """Service for ML model inference."""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._get_image_transform()
        self.stress_classes = ['healthy', 'drought', 'fungal', 'unknown']
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a dummy model for demonstration
            self.model = self._create_dummy_model()
    
    def _get_image_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration purposes."""
        class DummyModel(nn.Module):
            def __init__(self, num_classes=4):
                super(DummyModel, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = DummyModel(num_classes=len(self.stress_classes))
        model.to(self.device)
        model.eval()
        return model
    
    def load_model(self, model_path):
        """Load a trained model from file."""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = self._create_dummy_model()
    
    def preprocess_image(self, image_path):
        """Preprocess image for model inference."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_environmental_data(self, weather_data, soil_data):
        """Preprocess environmental data for model inference."""
        features = []
        
        # Weather features
        if weather_data:
            features.extend([
                weather_data.get('temperature', 0),
                weather_data.get('humidity', 0),
                weather_data.get('pressure', 0),
                weather_data.get('wind_speed', 0),
                weather_data.get('cloudiness', 0),
                weather_data.get('precipitation_1h', 0)
            ])
        else:
            features.extend([0] * 6)  # Default values
        
        # Soil features
        if soil_data:
            features.extend([
                soil_data.get('ph_water', 7.0),
                soil_data.get('organic_carbon', 0),
                soil_data.get('nitrogen', 0),
                soil_data.get('sand_content', 0),
                soil_data.get('clay_content', 0),
                soil_data.get('silt_content', 0)
            ])
        else:
            features.extend([7.0, 0, 0, 0, 0, 0])  # Default values
        
        # Normalize features (simple min-max scaling for demo)
        features = np.array(features, dtype=np.float32)
        
        # Convert to tensor
        return torch.tensor(features).unsqueeze(0).to(self.device)
    
    def predict(self, image_path, weather_data=None, soil_data=None):
        """
        Run inference on image and environmental data.
        
        Args:
            image_path (str): Path to the image file
            weather_data (dict): Weather data from external API
            soil_data (dict): Soil data from external API
            
        Returns:
            dict: Prediction results with stress scores
        """
        try:
            if not self.model:
                return None
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None
            
            # For this demo, we'll use only the image for prediction
            # In a real implementation, you would fuse image and environmental features
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # Create stress scores dictionary
            stress_scores = {}
            for i, class_name in enumerate(self.stress_classes):
                stress_scores[class_name] = float(probs[i])
            
            # Add some randomness for demo purposes (remove in real implementation)
            import random
            stress_scores = {
                'healthy': random.uniform(0.1, 0.9),
                'drought': random.uniform(0.0, 0.8),
                'fungal': random.uniform(0.0, 0.6),
                'unknown': random.uniform(0.0, 0.3)
            }
            
            # Normalize to sum to 1
            total = sum(stress_scores.values())
            stress_scores = {k: v/total for k, v in stress_scores.items()}
            
            return {
                'stress_scores': stress_scores,
                'dominant_stress': max(stress_scores, key=stress_scores.get),
                'confidence': max(stress_scores.values())
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

