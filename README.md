# ML Project — Numerical + Image datasets

This repository contains a small ML project using:

- Numerical dataset: `insurance.csv` (placed in project root)
- Image dataset: `image_dataset/` (YOLO-like labels under `train/labels`, `val/labels`, `test/labels`)

Structure

```
project/
├─ README.md
├─ requirements.txt
├─ setup.py
├─ insurance.csv
├─ image_dataset/
│  ├─ data.yaml
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ src/
│  ├─ data_loader.py        # functions for loading CSVs and parsing image labels
│  ├─ linear_model.py       # data cleaning, training, evaluation, saving sklearn model
│  └─ __init__.py
├─ scripts/
│  ├─ train_linear.py       # runs numerical pipeline end-to-end
│  └─ prepare_image_dataset.py
└─ tests/
   └─ test_linear.py
```

Quick start

1. Activate your virtualenv:

```bash
cd /home/ahmed/Work/Ahmed/ThirdYear/Cs/ML/project
source myenv/bin/activate
```

2. Install requirements (if not already):

```bash
pip install -r requirements.txt
```

3. Train the linear regression model on the numerical dataset:

```bash
python scripts/train_linear.py --data-path insurance.csv --output-dir models
```

4. Prepare / inspect the image dataset labels (example):

```bash
python scripts/prepare_image_dataset.py --dataset-dir image_dataset
```

Notes

- The repository is intended for uploading to GitHub. The `.gitignore` excludes virtualenvs and common artifacts.
- If you want reproducible pins, run `pip freeze > requirements.txt` after installing the desired versions.
