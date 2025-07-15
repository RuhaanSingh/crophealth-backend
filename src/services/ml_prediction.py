import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Define the Encoder class from the Kaggle notebook
class Encoder(nn.Module):
    """
    A lightweight CNN that encodes an image into a low-dimensional vector space.
    This is the core of the Prototypical Network.
    """
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
            # Final Pooling and Flattening
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # projection head to map features to the target embedding dimension
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        return self.projection(self.features(x))

class MLPredictionService:
    def __init__(self, model_path, prototypes_path, embedding_dim=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Encoder(embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load global prototypes
        self.global_prototypes = torch.load(prototypes_path, map_location=self.device)

        # These should match the classes from your training dataset
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        self.num_classes = len(self.class_names)
        self.embedding_dim = embedding_dim

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.4739, 0.4901, 0.4228], [0.2282, 0.2231, 0.2415])
        ])

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image_tensor)
            dists = torch.cdist(embedding, self.global_prototypes)
            preds = torch.argmin(dists, dim=1)
            predicted_class_index = preds.item()
            predicted_class_name = self.class_names[predicted_class_index]

            # Calculate confidence score (softmax of negative distances)
            confidence_scores = torch.nn.functional.softmax(-dists, dim=1)
            confidence = confidence_scores[0][predicted_class_index].item()

        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "stress_level": "high" if "healthy" not in predicted_class_name else "low"
        }

# Initialize the service with the model path and prototypes path
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "plant_disease_prototypical_final.pth")
prototypes_path = os.path.join(os.path.dirname(__file__), "..", "..", "global_prototypes.pt")
ml_service = MLPredictionService(model_path, prototypes_path)


