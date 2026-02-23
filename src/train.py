"""
Train a baseline classifier for phishing detection.
Supports Logistic Regression, Random Forest, or SVM.
Optionally saves the trained model.
"""
import os
import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from . import data
from . import utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_FILE = "model.joblib"


MODELS = {
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="rbf", random_state=42),
}


def train(
    model_name: str = "lr",
    data_path: str = None,
    save_model: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load data, train baseline model, optionally save it.

    Args:
        model_name: One of 'lr', 'rf', 'svm'
        data_path: Path to CSV dataset (optional)
        save_model: Whether to save model to results/
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Fitted model, and (X_train, X_test, y_train, y_test, feature_names)
    """
    X_train, X_test, y_train, y_test, feature_names, scaler = data.load_dataset(
        filepath=data_path, test_size=test_size, random_state=random_state
    )

    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {list(MODELS.keys())}")
    model = MODELS[model_name]

    model.fit(X_train, y_train)

    if save_model:
        out_dir = DEFAULT_MODEL_DIR
        utils.ensure_dir(out_dir)
        model_path = os.path.join(out_dir, MODEL_FILE)
        joblib.dump(
            {"model": model, "scaler": scaler, "feature_names": feature_names},
            model_path,
        )

    return model, (X_train, X_test, y_train, y_test, feature_names)


def main():
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="lr",
        help="Baseline model: lr, rf, or svm",
    )
    parser.add_argument("--data", default=None, help="Path to dataset CSV")
    parser.add_argument("--no-save", action="store_true", help="Do not save model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_path=args.data,
        save_model=not args.no_save,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(f"Training done. Model: {args.model}")


if __name__ == "__main__":
    main()
