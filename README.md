# ai-cyber-lab2-am4052

**CSEC 559/659 — Generative AI in Cybersecurity, Lab 2**  
Track 1: Phishing Detection

---

## 1. Project description

This project implements a **phishing detection** pipeline for binary classification of URLs (or URL-derived features) into **phishing** vs **benign**. It provides data loading and preprocessing, baseline classifiers (Logistic Regression, Random Forest, SVM), evaluation with standard metrics, and an exploratory data analysis (EDA) notebook. The goal is to establish a reproducible baseline that can be extended with better features or models.

**Repository layout:**

```
ai-cyber-lab2-am4052/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Your dataset CSV (e.g. phishing.csv)
│   └── processed/        # Processed / sample data
├── notebooks/
│   └── 01_eda.ipynb      # EDA: class distribution, feature summaries, insight
├── src/
│   ├── __init__.py
│   ├── data.py           # Load, clean, train/test split
│   ├── train.py          # Train baseline (LR / RF / SVM)
│   ├── eval.py           # Evaluate and save metrics + confusion matrix
│   └── utils.py          # Helpers
└── results/
    ├── metrics.json
    └── confusion_matrix.png
```

---

## 2. Dataset source and features

- **Source:** Raw data is **dataset_phishing.csv** (from `data/raw/dataset_phishing.csv.zip`). This is a public-style URL phishing dataset with one row per URL and 87 numeric features plus a `status` label (legitimate/phishing). A small **synthetic sample** remains in `data/processed/sample_phishing.csv` (800 rows) if you run without the raw file.
- **Format:** CSV with numeric feature columns; target is **status**: `legitimate` = benign (0), `phishing` = malicious (1). The `url` column is excluded from features.
- **Current dataset (dataset_phishing.csv):**  
  **11,430 samples**, **87 features**, **balanced classes** (5,715 legitimate, 5,715 phishing).  
  Features include URL structure (e.g. `length_url`, `length_hostname`, `ip`, `nb_dots`, `nb_slash`), character and word statistics (`ratio_digits_url`, `shortest_word_host`, `longest_word_path`), domain/page cues (`nb_hyperlinks`, `ratio_intHyperlinks`, `login_form`, `iframe`), and external signals (`whois_registered_domain`, `domain_age`, `google_index`, `page_rank`), among others. All are numeric; no encoding beyond scaling is applied.

---

## 3. Installation instructions

1. Clone or download this repository and `cd` into the project directory.
2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Place your dataset in `data/raw/` (e.g. unzip `dataset_phishing.csv.zip` to get `dataset_phishing.csv`). If no raw file is present, the code falls back to `data/processed/sample_phishing.csv`.

---

## 4. Training and evaluation commands

**Training** (default: Logistic Regression; optional: Random Forest or SVM):

```bash
python -m src.train --model lr
# Or: --model rf | --model svm
# Custom data: python -m src.train --model lr --data data/raw/phishing.csv
```

**Evaluation** (uses the last saved model and the same data split):

```bash
python -m src.eval
```

Outputs are written to `results/metrics.json` and `results/confusion_matrix.png`.

**EDA:** Open and run all cells in `notebooks/01_eda.ipynb` for class distribution, feature summaries, and a short written insight.

---

## 5. Baseline results

Results below are from **dataset_phishing.csv** (11,430 samples, 87 features) and the **Logistic Regression** baseline (default seed 42, 80/20 stratified train/test split).

| Metric    | Value   |
|----------|--------|
| Accuracy | **0.936** |
| Precision| **0.939** |
| Recall   | **0.933** |
| F1-score | **0.936** |

The balanced, feature-rich URL dataset yields strong baseline performance. Further gains may come from other models (e.g. Random Forest, SVM), feature selection, or hyperparameter tuning.

---

## 6. Ethics and safety consideration

- **Purpose:** This project is for educational and research use in a controlled course/lab setting. Phishing detectors can help protect users but must be designed and deployed responsibly.
- **Bias and fairness:** Models may reflect biases in the training data (e.g. over-representation of certain domains or languages). Care should be taken to avoid disproportionate false positives or negatives for particular groups or legitimate sites.
- **Misuse:** The same methods could be misused to evade detection or to target individuals. The code and models are intended for defensive use only (e.g. filtering or flagging suspicious URLs in a security product), not for crafting or distributing phishing content.
- **Data and privacy:** If you use datasets containing real URLs or user data, ensure you comply with their licenses, terms of use, and applicable privacy regulations. Do not commit sensitive or personal data to the repository.

---

## License

Course project; use per course and institutional policies.
