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
