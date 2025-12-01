import os
import cv2
import csv
import glob
import numpy as np
from skimage.feature import hog

# ================= CONFIGURATION =================
# DATASET_ROOT = 'tomato_dataset' 
OUTPUT_CSV = 'tomato_objects_hog.csv'

# We resize individual TOMATO CROPS to this size before HOG.
# 64x64 is sufficient for a single object and keeps features low.
CROP_SIZE = (64, 64) 

HOG_OPTS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8), # Smaller cells because our crop is small (64x64)
    'cells_per_block': (2, 2),
    'visualize': False,
    'channel_axis': None 
}

# ================= HELPERS =================

def yolo_to_pixels(yolo_line, img_h, img_w):
    """
    Converts YOLO format (class, x_center, y_center, width, height) 
    to pixel coordinates (x1, y1, x2, y2).
    """
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    x_c = float(parts[1])
    y_c = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    # Convert normalized to pixels
    x_c_pix = x_c * img_w
    y_c_pix = y_c * img_h
    w_pix = w * img_w
    h_pix = h * img_h

    # Calculate corners
    x1 = int(x_c_pix - (w_pix / 2))
    y1 = int(y_c_pix - (h_pix / 2))
    x2 = int(x_c_pix + (w_pix / 2))
    y2 = int(y_c_pix + (h_pix / 2))

    # Clamp values to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return class_id, x1, y1, x2, y2

def map_label(class_id):
    """
    Your custom logic:
    0, 1 -> Fresh (0)
    2, 3 -> Rotten (1)
    Other -> None (Ignore)
    """
    if class_id == 0 or class_id == 1:
        return 0 # Fresh
    elif class_id == 2 or class_id == 3:
        return 1 # Rotten
    return None

# ================= MAIN PROCESS =================

def process_dataset(DATASET_ROOT):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header_written = False
        
        objects_processed = 0
        images_processed = 0

        # Iterate through splits
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(DATASET_ROOT, split, 'images')
            lbl_dir = os.path.join(DATASET_ROOT, split, 'labels')
            
            # Find images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(img_dir, ext)))

            for img_path in image_files:
                # 1. READ LABEL FILE
                basename = os.path.basename(img_path)
                name_no_ext = os.path.splitext(basename)[0]
                txt_path = os.path.join(lbl_dir, name_no_ext + '.txt')

                if not os.path.exists(txt_path):
                    continue

                # Read lines first to see if we need to load the image at all
                with open(txt_path, 'r') as lf:
                    lines = lf.readlines()
                
                if not lines:
                    continue

                # 2. LOAD IMAGE (Grayscale)
                # We load it once per file
                img = cv2.imread(img_path, 0)
                if img is None: continue
                
                h, w = img.shape

                # 3. PROCESS EACH OBJECT IN THE IMAGE
                for line in lines:
                    try:
                        # Get coordinates
                        class_id, x1, y1, x2, y2 = yolo_to_pixels(line, h, w)
                        
                        # Apply Label Mapping
                        mapped_label = map_label(class_id)
                        if mapped_label is None:
                            continue # Skip unknown classes

                        # Safety check for empty crops
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # CROP the specific tomato
                        crop = img[y1:y2, x1:x2]
                        
                        # Resize crop
                        crop_resized = cv2.resize(crop, CROP_SIZE)

                        # HOG on the crop
                        features = hog(crop_resized, **HOG_OPTS)

                        # Write Header (once)
                        if not header_written:
                            header = [f'f{i}' for i in range(len(features))] + ['label']
                            writer.writerow(header)
                            header_written = True
                            print(f"Feature vector size per object: {len(features)}")

                        # Write Row
                        row = list(features) + [mapped_label]
                        writer.writerow(row)
                        objects_processed += 1
                        
                    except Exception as e:
                        # Handle malformed lines
                        continue

                # Explicit cleanup
                del img
                images_processed += 1
                
                if images_processed % 500 == 0:
                    print(f"Processed {images_processed} images ({objects_processed} tomato objects extracted)...")

    print(f"Done. Processed {images_processed} images.")
    print(f"Extracted {objects_processed} individual tomatoes to {OUTPUT_CSV}.")

if __name__ == "__main__":
    process_dataset()