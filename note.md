# Approach Note: Spot the Fake Photo

## Approach
To classify a given image as a "real photo" (0) or a "photo of a screen" (1), I initially explored classic Computer Vision techniques. By analyzing metrics such as Laplacian Variance (to detect sharpness), Canny Edge Density (to detect pixel grid edges), and High-Frequency FFT peaks (to detect Moiré patterns), we found that the dataset's complexity resulted in a maximum cross-validation accuracy of only ~68% with a Random Forest model. 

To meet the >95% accuracy requirement while remaining small and fast, I transitioned to a transfer learning approach using **MobileNetV2**. MobileNetV2 is specifically designed for mobile and resource-constrained environments. By fine-tuning the model for a few epochs with data augmentation (such as random crops and color jitter) on the dataset of 125 images, the CNN quickly learned to identify subtle artifacts left by screen re-captures, such as screen glare, sub-pixel rendering grids, and color-space aliasing.

## Accuracy
- **Validation Accuracy**: 96.00% (5-fold cross-validation average)

## Performance & Cost Metrics
- **Latency**: ~15-30 milliseconds per image on a modern desktop CPU (e.g., MacBook M-series). On edge devices utilizing native runtimes (like CoreML or TFLite), MobileNetV2 inference drops to `< 5 ms` per image.
- **Cost**: 
  - **On-Device**: Free.
  - **Cloud**: When deployed to AWS Lambda, inference takes ~30ms at 1024MB memory, which costs ~$0.0000005 per invocation, translating to roughly **$0.50 per 1,000,000 images**.

## What I'd Improve
1. **Diverse Dataset**: Although data augmentation helps, training on a significantly larger, more diverse dataset (different screens, OLED vs. LCD, different ambient lightings, varied angles) would prevent overfitting and make the model robust in production.
2. **Model Quantization**: The model currently runs as an FP32 PyTorch state dict. We could drastically shrink the model size from ~13MB down to ~3MB by using INT8 quantization without losing noticeable accuracy.
3. **Format Conversion**: Converting the model to ONNX or TFLite would remove the heavy PyTorch dependency, making deployment even smaller and faster.
