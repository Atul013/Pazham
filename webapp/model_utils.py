# Function to make predictions from image
import torch
import os
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# Define model at module level
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    global model
    if model is None:
        model = BananaNet().to(device)
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

# Original function, renamed for flexibility
def predict_from_image_full(model, image_path):
    """
    Predict seed count and curvature from a banana image
    Args:
        model: Trained BananaNet model
        image_path: Path to the banana image
    Returns:
        dict: Predictions for seed count and curvature
    """
    image_tensor = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
        seed_count = int(round(predictions[0][0].item()))
        curvature = round(predictions[0][1].item(), 1)
        return {
            'seeds': seed_count,
            'curvature': curvature
        }

# Wrapper for compatibility with app.py
def predict_from_image(image_path):
    global model
    load_model()  # Ensure model is loaded
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
        seed_count = int(round(predictions[0][0].item()))
        curvature = round(predictions[0][1].item(), 1)
        return {
            'seeds': seed_count,
            'curvature': curvature
        }

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


class BananaNet(nn.Module):
    def __init__(self):
        super(BananaNet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.base_model(x)
        output = self.regression_head(features)
        return output


# Load model and weights
model = BananaNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Loaded trained model weights from best_model.pth")
else:
    print("Warning: best_model.pth not found, using untrained model.")
model.eval()
