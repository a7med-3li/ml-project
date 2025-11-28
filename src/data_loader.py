"""Data loading helpers for numerical CSVs and simple parsing for the image dataset.

This module provides:
- load_csv(path): returns a pandas DataFrame
- prepare_image_label_dataframe(dataset_dir): for a YOLO-like dataset (images + labels .txt)
"""
from typing import List
import os
import pandas as pand
import glob


def load_csv(path: str) -> pand.DataFrame:
    """Load a CSV into a DataFrame.

    Args:
        path: path to CSV file

    Returns:
        pandas DataFrame
    """
    if not os.path.exists(path):
        print
        raise FileNotFoundError(f"CSV not found: {path}")
    return pand.read_csv(path)


def _read_label_file(label_path: str) -> List[int]:
    """Read a YOLO-style label file and return list of class indices found (may be empty).

    Format: each line `class x_center y_center width height`
    """
    classes = []
    if not os.path.exists(label_path):
        return classes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    classes.append(int(float(parts[0])))
                except Exception:
                    continue
    return classes


def prepare_image_label_dataframe(dataset_dir: str):
    """Prepare a DataFrame with image path and primary class label inferred from label files.

    Expects dataset_dir to contain `train/images`, `train/labels`, `val/` and `test/` similarly.
    For each image, looks for a corresponding .txt label file in the parallel labels directory.
    """
    rows = []
    for split in ('train', 'val', 'test'):
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        if not os.path.isdir(images_dir):
            continue
        for img_path in glob.glob(os.path.join(images_dir, '*')):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            # many label files in this repo have names with additional suffixes; try common patterns
            possible = [
                os.path.join(labels_dir, img_name + '.txt'),
                os.path.join(labels_dir, img_name + '.rf.txt'),
            ]
            # also try any label file that starts with the base name
            matched_label = None
            for p in possible:
                if os.path.exists(p):
                    matched_label = p
                    break
            if matched_label is None:
                # fallback: try to find a label file that starts with img_name
                candidates = glob.glob(os.path.join(labels_dir, img_name + '*'))
                if candidates:
                    matched_label = candidates[0]

            classes = _read_label_file(matched_label) if matched_label else []
            label = classes[0] if classes else None
            rows.append({
                'split': split,
                'image_path': img_path,
                'label': label,
                'label_file': matched_label,
            })
    df = pand.DataFrame(rows)
    return df
