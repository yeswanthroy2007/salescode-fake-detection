import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def get_chroma_fft_peaks(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None
    
    img = img.resize((256, 256))
    img = np.array(img)
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:,:,1]
    cb = ycrcb[:,:,2]
    
    peaks = 0
    for channel in (cr, cb):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        cv2.circle(mag, (cx, cy), 20, 0, -1)
        
        peaks += np.sum(mag > (np.mean(mag) + 3*np.std(mag)))
        
    return peaks

def main():
    real_paths = glob("Dataset/Real/*.*")
    screen_paths = glob("Dataset/Screen/*.*")
    
    real_peaks = []
    for p in real_paths:
        res = get_chroma_fft_peaks(p)
        if res is not None: real_peaks.append(res)
            
    screen_peaks = []
    for p in screen_paths:
        res = get_chroma_fft_peaks(p)
        if res is not None: screen_peaks.append(res)
        
    print("Real Peaks:", sorted(real_peaks))
    print("Screen Peaks:", sorted(screen_peaks))
    
    best_acc = 0
    best_thresh = 0
    for thresh in range(0, 100):
        # Predict screen if peaks > thresh 
        correct_real = sum(1 for p in real_peaks if p <= thresh)
        correct_screen = sum(1 for p in screen_peaks if p > thresh)
        
        acc = (correct_real + correct_screen) / (len(real_peaks) + len(screen_peaks))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    print(f"Chroma FFT Best Accuracy: {best_acc:.4f} at threshold {best_thresh}")

if __name__ == '__main__':
    main()
