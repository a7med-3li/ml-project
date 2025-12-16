import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def read_yolo_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append((class_id, x_center, y_center, width, height))
    return labels

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x_min = int(x_center_abs - width_abs / 2)
    y_min = int(y_center_abs - height_abs / 2)
    x_max = int(x_center_abs + width_abs / 2)
    y_max = int(y_center_abs + height_abs / 2)
    
    return x_min, y_min, x_max, y_max

def crop_tomato(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def resize_image(image, target_size=(64, 64)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def image_to_feature_vector(image):
    return image.flatten()

def create_tomato_csv(images_dir, labels_dir, output_csv, target_size=(64, 64), batch_size=50):
    # Class names mapping
    class_names = {
        0: '0',
        1: '1',
        2: '2',
        3: '3'
    }

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images to process...")
    
    # Create column names
    num_pixels = target_size[0] * target_size[1] * 3
    columns = ['image_name', 'tomato_id'] + [f'pixel_{i}' for i in range(num_pixels)] + ['label']
    
    # Write header to CSV
    with open(output_csv, 'w') as f:
        f.write(','.join(columns) + '\n')
    
    data_batch = []
    total_tomatoes = 0
    label_counts = {}
    
    for img_idx, img_file in enumerate(image_files):
        # Construct paths
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_file}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        img_height, img_width = image.shape[:2]

        labels = read_yolo_labels(label_path)
        
        # Process each tomato in the image
        for tomato_id, (class_id, x_center, y_center, width, height) in enumerate(labels):
        
            bbox = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            
            cropped_tomato = crop_tomato(image, bbox)
            
            # Skip if crop is too small or invalid
            if cropped_tomato.size == 0 or cropped_tomato.shape[0] < 5 or cropped_tomato.shape[1] < 5:
                continue
            
            resized_tomato = resize_image(cropped_tomato, target_size)
            
            feature_vector = image_to_feature_vector(resized_tomato)
            
            label = class_names.get(class_id, f'class_{class_id}')
            label_counts[label] = label_counts.get(label, 0) + 1
            
            row = [img_file, tomato_id] + feature_vector.tolist() + [label]
            data_batch.append(row)
            total_tomatoes += 1
            
            # Write batch to disk when it reaches batch_size
            if len(data_batch) >= batch_size:
                df_batch = pd.DataFrame(data_batch, columns=columns)
                df_batch.to_csv(output_csv, mode='a', header=False, index=False)
                data_batch = []  # Clear batch from memory
                print(f"Processed {total_tomatoes} tomatoes...")
        
        # Clear image from memory after processing
        del image
        
        if (img_idx + 1) % 50 == 0:
            print(f"Processed {img_idx + 1}/{len(image_files)} images...")
    
    # Write remaining data
    if data_batch:
        df_batch = pd.DataFrame(data_batch, columns=columns)
        df_batch.to_csv(output_csv, mode='a', header=False, index=False)
    
    print(f"\nCSV file created successfully!")
    print(f"Total tomatoes extracted: {total_tomatoes}")
    print(f"Features per tomato: {num_pixels}")
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"Output saved to: {output_csv}")