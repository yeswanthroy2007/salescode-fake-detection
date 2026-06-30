import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pillow_heif import register_heif_opener
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

register_heif_opener()

def get_features(model, preprocess, img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(input_batch)
    return features.squeeze().numpy()

def main():
    model = models.mobilenet_v2(pretrained=True)
    # Remove the classifier head, we just want features
    model.classifier = torch.nn.Identity()
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    real_paths = glob("Dataset/Real/*.*")
    screen_paths = glob("Dataset/Screen/*.*")
    
    X = []
    y = []
    
    print(f"Extracting features for {len(real_paths)} real images...")
    for p in real_paths:
        f = get_features(model, preprocess, p)
        if f is not None:
            X.append(f)
            y.append(0)
            
    print(f"Extracting features for {len(screen_paths)} screen images...")
    for p in screen_paths:
        f = get_features(model, preprocess, p)
        if f is not None:
            X.append(f)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    
    print(f"Cross-Validation Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    # Let's save the model
    clf.fit(X, y)
    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model saved to model.pkl")

if __name__ == '__main__':
    main()
