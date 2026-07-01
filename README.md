# Screen Detection: Real Photo vs Screen Capture

This project is a lightweight deep learning model that classifies an input image as either a **real photograph** or a **photo of a screen** (re-captured image).

The model is built using a fine-tuned **MobileNetV2** architecture and delivers **96.00% classification accuracy** on the evaluation dataset while maintaining fast inference and a compact model size.

---

##  Installation

The project is implemented using **PyTorch** and **Torchvision**. Creating a virtual environment is recommended before installing the required packages.

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

### 2. Install the required dependencies

```bash
pip install torch torchvision pillow pillow-heif
```

> **Note:** `pillow-heif` enables support for `.HEIC` images commonly produced by iPhones.

---

##  How to Use

### Single Image Prediction

Run the following command to classify a single image. The script returns a probability score between:

- **0.0** → Real Photo
- **1.0** → Photo of a Screen

```bash
python predict.py "path/to/image.jpg"
```

**Example**

```bash
python predict.py "Dataset/Screen/fake1.jpeg"
```

Example Output:

```text
0.8138120174407959
```

---

### Batch Prediction

Use `batch_inference.py` to process an entire folder of images. The script generates a CSV report containing prediction scores and can optionally organize the images based on their predicted class.

#### CSV Report Only

```bash
python batch_inference.py "path/to/input_folder" "results.csv"
```

#### CSV Report + Image Sorting

```bash
python batch_inference.py "path/to/input_folder" "results.csv" "path/to/output_folder"
```

The output folder will contain:

- `Real/` – Images predicted as genuine photographs.
- `Screen/` – Images predicted as screen captures.

---

##  Project Structure

- **predict.py** – Performs inference on a single image.
- **batch_inference.py** – Processes multiple images and generates prediction logs.
- **best_model.pth** – Trained MobileNetV2 model weights.
- **note.md** – Summary of the project, methodology, and evaluation results.
- **research-work/** – Development scripts used for dataset analysis, experimentation, model training, and evaluation.

---

##  Model Performance

- **Model:** MobileNetV2 (Fine-tuned)
- **Task:** Binary Image Classification
- **Classes:** Real Photo / Screen Capture
- **Evaluation Accuracy:** **97.00%**
- **Framework:** PyTorch
