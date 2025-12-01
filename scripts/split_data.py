import csv
import random
import os

# ================= CONFIGURATION =================
INPUT_CSV = '../assets/image_dataset/tomato_objects_hog.csv'  # The big file you just made
TRAIN_CSV = 'train_data.csv'
TEST_CSV = 'test_data.csv'

# Split ratio: 0.8 means 80% train, 20% test
SPLIT_RATIO = 0.8 
# =================================================

def split_csv_streaming():
    print(f"Reading from: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print("Starting data split...")
    
    # Open all 3 files at once
    with open(INPUT_CSV, 'r') as f_in, \
         open(TRAIN_CSV, 'w', newline='') as f_train, \
         open(TEST_CSV, 'w', newline='') as f_test:
        
        reader = csv.reader(f_in)
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        # 1. Handle Header
        try:
            header = next(reader)
            # Write the header to both new files
            writer_train.writerow(header)
            writer_test.writerow(header)
        except StopIteration:
            print("Error: CSV file is empty.")
            return

        # 2. Process rows line by line
        train_count = 0
        test_count = 0

        for row in reader:
            # Generate a random number between 0.0 and 1.0
            if random.random() < SPLIT_RATIO:
                writer_train.writerow(row)
                train_count += 1
            else:
                writer_test.writerow(row)
                test_count += 1

    print("=======================================")
    print("Split Complete!")
    print(f"Training samples: {train_count}")
    print(f"Testing samples:  {test_count}")
    print(f"Files saved: '{TRAIN_CSV}' and '{TEST_CSV}'")

if __name__ == "__main__":
    # Optional: Set seed for reproducibility (so you get the same split every time)
    random.seed(42) 
    split_csv_streaming()