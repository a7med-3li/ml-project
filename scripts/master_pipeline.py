import os
import cv2
import csv
import glob
import numpy as np
import pandas as pd
from skimage.feature import hog

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= CONFIGURATION =================
DATASET_ROOT = '../assets/image_dataset/train' 
CSV_FILE = 'tomato_final_data.csv'

# Image Processing
CROP_SIZE = (64, 64) 
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'visualize': False,
    'channel_axis': None
}

# Machine Learning Settings
PCA_COMPONENTS = 50  # Compress 1700 features down to 50 best features
TEST_SPLIT = 0.2
SEED = 42

# ================= PART 1: DATA ENGINEERING =================

def yolo_to_coords(line, img_h, img_w):
    parts = line.strip().split()
    cls = int(parts[0])
    x_c, y_c, w, h = map(float, parts[1:])
    
    x1 = int((x_c - w/2) * img_w)
    y1 = int((y_c - h/2) * img_h)
    x2 = int((x_c + w/2) * img_w)
    y2 = int((y_c + h/2) * img_h)
    return cls, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def map_label(cls):
    if cls in [0, 1]: return 0 # Fresh
    if cls in [2, 3]: return 1 # Rotten
    return None

def generate_dataset_if_needed():
    if os.path.exists(CSV_FILE):
        print(f"[INFO] {CSV_FILE} already exists. Skipping image processing.")
        return

    print("[INFO] Generating dataset from images (This happens once)...")
    
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header_written = False
        count = 0
        
        # Search all subfolders
        all_images = glob.glob(os.path.join(DATASET_ROOT, '**', '*.jpg'), recursive=True) + \
                     glob.glob(os.path.join(DATASET_ROOT, '**', '*.png'), recursive=True)

        for img_path in all_images:
            # Derive label path
            label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            
            if not os.path.exists(label_path): continue
            
            # Load Image (Grayscale)
            img = cv2.imread(img_path, 0)
            if img is None: continue
            h, w = img.shape
            
            with open(label_path, 'r') as lf:
                lines = lf.readlines()
                
            for line in lines:
                try:
                    cls, x1, y1, x2, y2 = yolo_to_coords(line, h, w)
                    label = map_label(cls)
                    if label is None or x2<=x1 or y2<=y1: continue
                    
                    # Crop & HOG
                    crop = cv2.resize(img[y1:y2, x1:x2], CROP_SIZE)
                    features = hog(crop, **HOG_PARAMS)
                    
                    if not header_written:
                        writer.writerow([f'f{i}' for i in range(len(features))] + ['label'])
                        header_written = True
                        
                    writer.writerow(list(features) + [label])
                    count += 1
                except: continue
                
            if count % 1000 == 0: print(f"   Extracted {count} tomatoes...")
            
    print(f"[SUCCESS] Saved {count} rows to {CSV_FILE}")

# ================= PART 2: MACHINE LEARNING =================

def run_ml_pipeline():
    print("\n[STEP 1] Loading Data...")
    df = pd.read_csv(CSV_FILE)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=SEED)
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Preprocessing Pipeline (Scaling + PCA)
    print(f"\n[STEP 2] Preprocessing (Scaling + PCA to {PCA_COMPONENTS} dims)...")
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS)
    
    # Fit on TRAIN, transform both
    X_train_proc = pca.fit_transform(scaler.fit_transform(X_train))
    X_test_proc = pca.transform(scaler.transform(X_test))
    
    # --- MODEL A: LOGISTIC REGRESSION ---
    print("\n[STEP 3] Training Logistic Regression (Supervised)...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_proc, y_train)
    
    lr_pred = lr.predict(X_test_proc)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    print(f"   >>> Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))

    # --- MODEL B: K-MEANS ---
    print("\n[STEP 4] Training K-Means (Unsupervised)...")
    km = KMeans(n_clusters=2, random_state=SEED, n_init='auto')
    km.fit(X_train_proc) # Train on PCA data
    
    # Predict on test set
    km_pred = km.predict(X_test_proc)
    
    # Calculate Accuracy (checking flipped labels)
    acc_normal = accuracy_score(y_test, km_pred)
    acc_flipped = accuracy_score(y_test, 1 - km_pred)
    
    if acc_normal > acc_flipped:
        final_km_acc = acc_normal
        best_km_pred = km_pred
        mapping = "Cluster 0 = Fresh"
    else:
        final_km_acc = acc_flipped
        best_km_pred = 1 - km_pred
        mapping = "Cluster 0 = Rotten"
        
    print(f"   >>> K-Means Accuracy: {final_km_acc*100:.2f}%")
    print(f"   (Logic used: {mapping})")
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, best_km_pred))

    # --- CONCLUSION ---
    print("\n===========================================")
    print("FINAL COMPARISON")
    print("===========================================")
    print(f"Logistic Regression: {lr_acc*100:.2f}% (Uses Labels)")
    print(f"K-Means Clustering:  {final_km_acc*100:.2f}% (No Labels)")
    
    if lr_acc > final_km_acc + 0.10:
        print("\nInsight: The Supervised model is significantly better.")
        print("This means the difference between Rotten and Fresh is subtle")
        print("and requires label guidance to learn effectively.")
    else:
        print("\nInsight: K-Means performed surprisingly well!")
        print("This means the HOG features + PCA created very distinct")
        print("visual clusters naturally.")

if __name__ == "__main__":
    # 1. Prepare Data
    generate_dataset_if_needed()
    
    # 2. Run Models
    run_ml_pipeline()