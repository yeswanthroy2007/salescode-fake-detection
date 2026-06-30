# Screen Detector: Spot the Fake Photo

This project is a lightweight, high-accuracy machine learning classifier designed to determine whether a given image is a genuine "real photo" or a "photo of a screen" (a re-capture/fraud). 

It achieves **96.00% accuracy** on the evaluation dataset by using a fine-tuned MobileNetV2 architecture.

---

## 🛠 Setup Instructions

This project relies on PyTorch and Torchvision. It is recommended to use a virtual environment.

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision pillow pillow-heif
   ```
   *(Note: `pillow-heif` is used to seamlessly support `.HEIC` photos taken from iOS devices).*

---

## 🚀 Usage

### 1. Single Image Prediction
You can predict the fraud probability of a single image using `predict.py`. It outputs a single float value between `0.0` (Real Photo) and `1.0` (Photo of a Screen).

**Command:**
```bash
python predict.py "path/to/image.jpg"
```

**Example:**
```bash
$ python predict.py "Dataset/Screen/fake1.jpeg"
0.8138120174407959
```

### 2. Batch Inference & Logging
To evaluate a large folder of images, use `batch_inference.py`. It can generate a CSV log of all scores, and optionally sort/copy the images into `Real` and `Screen` subfolders.

**Logging-only Mode:**
Iterates over a folder and logs the predictions to a CSV file.
```bash
python batch_inference.py "path/to/input_folder" "results.csv"
```

**Sorting & Logging Mode:**
Logs the results to CSV and copies the images into classified subdirectories (`output_folder/Real` and `output_folder/Screen`).
```bash
python batch_inference.py "path/to/input_folder" "results.csv" "path/to/output_folder"
```

---

## 📂 Project Structure

- `predict.py`: Core inference script for single images.
- `batch_inference.py`: Utility script for batch processing and CSV logging.
- `best_model.pth`: The fine-tuned MobileNetV2 PyTorch weights (~13MB).
- `note.md`: The approach report detailing accuracy, latency, and costs.
- `research/`: Contains all exploratory scripts (data analysis, classic CV testing, and fine-tuning scripts) used during the development phase.
