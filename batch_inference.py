import os
import sys
import shutil
import csv
from glob import glob
from predict import predict

def run_batch_inference(input_folder: str, log_file: str, output_folder: str = None):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Set up output folder for sorting if provided
    real_out, screen_out = None, None
    if output_folder:
        real_out = os.path.join(output_folder, "Real")
        screen_out = os.path.join(output_folder, "Screen")
        os.makedirs(real_out, exist_ok=True)
        os.makedirs(screen_out, exist_ok=True)
    
    # Supported image extensions (including HEIC since our predict script supports it)
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.heic')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob(os.path.join(input_folder, ext.upper())))
        
    print(f"Found {len(image_paths)} images in '{input_folder}'.")
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Fraud_Score", "Predicted_Class"])
        
        for i, img_path in enumerate(image_paths, 1):
            filename = os.path.basename(img_path)
            try:
                # Get prediction (0 to 1, where 1 is Screen)
                score = predict(img_path)
                
                if score >= 0.5:
                    pred_class = "Screen"
                    dest_path = os.path.join(screen_out, filename) if screen_out else None
                else:
                    pred_class = "Real"
                    dest_path = os.path.join(real_out, filename) if real_out else None
                    
                # Copy to respective folder only if output_folder is configured
                if dest_path:
                    shutil.copy2(img_path, dest_path)
                
                # Log to CSV
                writer.writerow([filename, f"{score:.4f}", pred_class])
                print(f"[{i}/{len(image_paths)}] {filename} -> {pred_class} (Score: {score:.4f})")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                writer.writerow([filename, "ERROR", str(e)])

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:")
        print("  Logging-only:  python batch_inference.py <input_folder> <log_file.csv>")
        print("  Logging & Copying: python batch_inference.py <input_folder> <log_file.csv> <output_folder>")
        sys.exit(1)
        
    in_dir = sys.argv[1]
    log_csv = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) == 4 else None
    
    run_batch_inference(in_dir, log_csv, out_dir)
    if out_dir:
        print(f"\nDone! Results saved to '{out_dir}' and logs written to '{log_csv}'.")
    else:
        print(f"\nDone! Logs written to '{log_csv}'.")
