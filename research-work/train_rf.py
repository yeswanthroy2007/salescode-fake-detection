import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

register_heif_opener()

def get_features(img_path):
    try:
        img = Image.open(img_path).convert('L')
    except Exception as e:
        return None
    
    img = img.resize((256, 256))
    gray = np.array(img)
    
    # 1. Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (256*256)
    
    # 3. FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    cv2.circle(mag, (cx, cy), 20, 0, -1)
    
    peaks = np.sum(mag > (np.mean(mag) + 3*np.std(mag)))
    
    # 4. Color variance (if we loaded RGB, but we loaded L). Let's load RGB for color var
    try:
        img_rgb = Image.open(img_path).convert('RGB')
        img_rgb = img_rgb.resize((256, 256))
        rgb = np.array(img_rgb)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        cr_var = ycrcb[:,:,1].var()
        cb_var = ycrcb[:,:,2].var()
    except Exception:
        cr_var = 0
        cb_var = 0
        
    return [lap_var, edge_density, peaks, cr_var, cb_var]

def main():
    real_paths = glob("Dataset/Real/*.*")
    screen_paths = glob("Dataset/Screen/*.*")
    
    X = []
    y = []
    
    for p in real_paths:
        res = get_features(p)
        if res is not None:
            X.append(res)
            y.append(0)
            
    for p in screen_paths:
        res = get_features(p)
        if res is not None:
            X.append(res)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    
    print(f"Random Forest CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    clf.fit(X, y)
    joblib.dump(clf, "rf_model.pkl")

if __name__ == '__main__':
    main()
