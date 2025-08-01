import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Define the CNN model
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

# Initialize model
model = BananaNet()
model.eval()

# Prediction function
def predict_from_image(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        preds = model(input_tensor)
        seed_count = int(round(preds[0][0].item()))
        curvature = round(preds[0][1].item(), 1)
    return {
        'seeds': max(0, seed_count),
        'curvature': max(20, min(70, curvature))
    }
