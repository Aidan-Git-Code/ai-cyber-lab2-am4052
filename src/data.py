"""
Load and preprocess phishing detection dataset.
Train/test split and return feature matrices and labels.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Default paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_dataset(
    filepath: str = None,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load dataset from file, clean, preprocess, and split into train/test.

    Expects CSV with numeric features and a binary label column (0=benign, 1=phishing).
    If filepath is None, looks for data/raw/phishing.csv or data/processed/train.csv.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    if filepath is None:
        for name in ("phishing.csv", "dataset_phishing.csv", "url_dataset.csv", "train.csv"):
            candidate = os.path.join(DEFAULT_RAW_DIR, name)
            if os.path.isfile(candidate):
                filepath = candidate
                break
        if filepath is None:
            for name in ("train.csv", "sample_phishing.csv"):
                candidate = os.path.join(DEFAULT_PROCESSED_DIR, name)
                if os.path.isfile(candidate):
                    filepath = candidate
                    break
            if filepath is None:
                filepath = os.path.join(DEFAULT_PROCESSED_DIR, "train.csv")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"No dataset found. Place a CSV in data/raw/ (e.g. phishing.csv) "
                f"with numeric features and a '{label_col}' column (0=benign, 1=phishing)."
            )

    df = pd.read_csv(filepath)

    # Use "status" as label if dataset has it (e.g. legitimate/phishing)
    if label_col not in df.columns and "status" in df.columns:
        label_col = "status"

    # Basic cleaning: drop rows with missing target
    if label_col not in df.columns:
        raise ValueError(
            f"Dataset must contain label column '{label_col}'. Columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[label_col])

    # Ensure binary numeric labels (support string labels like legitimate/phishing)
    raw_labels = np.asarray(df[label_col])
    str_mapping = {"benign": 0, "phishing": 1, "legitimate": 0, "safe": 0}
    try:
        y = raw_labels.astype(int)
    except (ValueError, TypeError):
        y = np.array([str_mapping.get(str(v).lower(), 0) for v in raw_labels]).astype(int)
    else:
        if set(np.unique(y)) - {0, 1}:
            y = np.array([str_mapping.get(str(v).lower(), v) for v in raw_labels]).astype(int)

    # Feature matrix: all numeric columns except label
    exclude = [label_col, "url", "email", "id"]  # common non-feature columns
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != label_col and c not in exclude]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    X = df[feature_cols].fillna(0).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional: scale features for SVM/Logistic Regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_cols, scaler


def get_raw_data_path(filename: str = "phishing.csv") -> str:
    """Return path to a file in data/raw/."""
    return os.path.join(DEFAULT_RAW_DIR, filename)
