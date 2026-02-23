# ai-cyber-lab2-am4052

**CSEC 559/659 — Generative AI in Cybersecurity, Lab 2**  
Track 1: Phishing Detection (binary classification of URLs/emails into phishing vs. benign).

## Project structure

```
ai-cyber-lab2-am4052/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/           # Place your dataset CSV here (e.g. phishing.csv)
│   └── processed/     # Processed/sample data
├── notebooks/
│   └── 01_eda.ipynb   # EDA: class distribution, feature summaries, insight
├── src/
│   ├── __init__.py
│   ├── data.py        # Load, clean, split data
│   ├── train.py       # Train baseline (LR / RF / SVM)
│   ├── eval.py        # Evaluate, save metrics + confusion matrix
│   └── utils.py       # Helpers
└── results/
    ├── metrics.json         # Accuracy, precision, recall, F1
    └── confusion_matrix.png # Confusion matrix plot
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

- Place your CSV in `data/raw/` (e.g. `phishing.csv` or `url_dataset.csv`).
- Required: numeric feature columns and a **label** column (`0` = benign, `1` = phishing).
- Optional: a sample dataset is provided in `data/processed/sample_phishing.csv` so you can run the pipeline without your own file.

## Usage

**Train a baseline model** (default: Logistic Regression):

```bash
python -m src.train --model lr    # or rf, svm
python -m src.train --model rf --data data/raw/phishing.csv
```

**Evaluate** (uses the last saved model and the same data split):

```bash
python -m src.eval
```

Outputs: `results/metrics.json`, `results/confusion_matrix.png`.

**EDA:**

Open `notebooks/01_eda.ipynb` and run all cells for class distribution, feature summaries, and a short written insight.

## Baseline models

- `lr` — Logistic Regression  
- `rf` — Random Forest  
- `svm` — SVM (RBF kernel)

## Evaluation artifacts

- **results/metrics.json**: `accuracy`, `precision`, `recall`, `f1_score`
- **results/confusion_matrix.png**: Confusion matrix heatmap (Benign vs Phishing)

## License

Course project; use per course and institutional policies.
