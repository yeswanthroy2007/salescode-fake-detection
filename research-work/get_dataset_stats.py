import os
from glob import glob
from PIL import Image
from pillow_heif import register_heif_opener
import collections

register_heif_opener()

def analyze_folder(folder_path):
    paths = glob(os.path.join(folder_path, "*.*"))
    formats = collections.Counter()
    resolutions = []
    
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        formats[ext] += 1
        try:
            with Image.open(p) as img:
                resolutions.append(img.size)
        except Exception:
            pass
            
    num_files = len(paths)
    return num_files, formats, resolutions

def main():
    real_dir = "Dataset/Real"
    screen_dir = "Dataset/Screen"
    
    r_count, r_formats, r_res = analyze_folder(real_dir)
    s_count, s_formats, s_res = analyze_folder(screen_dir)
    
    print(f"--- Real Dataset Stats ---")
    print(f"Total Images: {r_count}")
    print(f"Formats: {dict(r_formats)}")
    if r_res:
        widths, heights = zip(*r_res)
        print(f"Resolution range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"Average Resolution: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
        
    print(f"\n--- Screen Dataset Stats ---")
    print(f"Total Images: {s_count}")
    print(f"Formats: {dict(s_formats)}")
    if s_res:
        widths, heights = zip(*s_res)
        print(f"Resolution range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"Average Resolution: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")

if __name__ == '__main__':
    main()
