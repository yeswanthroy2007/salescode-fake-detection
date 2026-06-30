"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Optional: Add pillow-heif if the user wants to test HEIC images.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Initialize model globally so it's only loaded once if `predict` is called multiple times.
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
        
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    if not os.path.exists(model_path):
        # Fallback to parent directory just in case
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pth")
        
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    _model = model
    return _model

def predict(image_path: str) -> float:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return 0.5 # Return uncertain score if image is unreadable
        
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    model = load_model()
    
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Class 1 is "photo of a screen"
    fraud_score = probabilities[1].item()
    return fraud_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)
    print(predict(sys.argv[1]))
