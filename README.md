# ML Project — Numerical + Image datasets

This repository contains example pipelines and notebooks for working with two datasets: a numerical insurance dataset and an image-based tomato-quality dataset.

- **Numerical dataset:** `assets/numerical_dataset/insurance.csv`
- **Image dataset (YOLO label format):** `assets/image_dataset/train/tomatoes_dataset.csv`
- here is its link: [tomatoes_dataset.csv](https://drive.google.com/drive/folders/19PVlE7EXqTCGUWgw7XhOujhAgr5_kUzp?usp=sharing)

**Repository Overview**

- `notebooks/`: Interactive analyses and experiments
  - `notebooks/numerical_dataset.ipynb` — regression pipeline and EDA
  - `notebooks/KNN_Regressor.ipynb` — KNN regression example
  - `notebooks/image_clasification.ipynb` — image preprocessing and classification/clustering experiments
- `scripts/`: utility scripts
  - `scripts/imageToCSV.py` — extract tomatoes from images/YOLO labels and create a CSV (batched writing to limit memory)
- `assets/`: dataset files (large generated CSVs are excluded from git)
- `requirements.txt`: runtime dependencies for the project

**Quick Start**

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv myenv
source myenv/bin/activate
```

2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Create the image-derived CSV (if you have `images/` and `labels/` folders):

```bash
python scripts/imageToCSV.py
# Default output: assets/image_dataset/train/tomatoes_dataset.csv
```

4. Open notebooks for interactive exploration:

```bash
jupyter lab  # or jupyter notebook
```

**Notes & recommendations**

- Large generated CSVs (e.g. `train_data.csv`, `test_data.csv`) are intentionally not tracked by git. Use artifact storage or Git LFS for versioning large files.
- `requirements.txt` lists runtime libraries used by the project; create a full pinned freeze inside your venv for exact reproducibility:

```bash
pip freeze > requirements.txt
```

If you want, I can run notebooks end-to-end and report numeric metrics, scaffold a small CLI (e.g. `scripts/train_linear.py`), or add tests.

---

Updated to reflect current scripts and notebooks.

# ML Project — Numerical + Image datasets

This repository contains a small ML project with two example datasets and simple training utilities.

- Numerical dataset: `assets/numerical_dataset/insurance.csv`
- Image dataset: `assets/image_dataset/` (YOLO-like labels under `train/labels`, `val/labels`, `test/labels`)

Repository structure (high level)

```
project/
├─ README.md
├─ requirements.txt
├─ setup.py
├─ assets/
│  ├─ numerical_dataset/
│  │  └─ insurance.csv
│  └─ image_dataset/
├─ src/                 # reusable library code
├─ scripts/             # runnable helpers and CLI entrypoints
├─ notebooks/           # experiments and exploratory analysis
└─ tests/
```

Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv myenv
source myenv/bin/activate
```

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

3. Jupyter notebooks

- Open the notebooks for interactive work (recommended):

```bash
jupyter lab  # or jupyter notebook
```

- Notebooks are under `notebooks/`, the main numerical pipeline is in `notebooks/numerical_dataset.ipynb`.

4. (Optional) Run a CLI training script

- There is a planned CLI at `scripts/train_linear.py` which will orchestrate loading, preprocessing and training. If present it can be run like:

```bash
python scripts/train_linear.py --data-path assets/numerical_dataset/insurance.csv --output-dir artifacts
```

Notes and tips

- Data files were moved into `assets/` to keep the repository root clean. Use `assets/numerical_dataset/insurance.csv` as the canonical path.
- If you need reproducible pins, update `requirements.txt` with exact versions and recreate your venv.
- Tests are under `tests/`. Run them with pytest after installing the dev deps.

Contributing

- Consider creating a branch and opening a PR for larger changes (CI/workflow additions, new models).

If you'd like, I can also scaffold `scripts/train_linear.py` and a small pytest harness so the CLI and tests are runnable.
