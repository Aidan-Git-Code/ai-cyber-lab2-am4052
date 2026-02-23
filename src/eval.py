"""
Evaluate phishing detection model on test set.
Compute accuracy, precision, recall, F1; save metrics.json and confusion matrix plot.
"""
import os
import argparse
import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _USE_SEABORN = True
except ImportError:
    _USE_SEABORN = False

from . import data
from . import utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_FILE = "metrics.json"
CONFUSION_PLOT = "confusion_matrix.png"
MODEL_FILE = "model.joblib"


def evaluate(
    model=None,
    X_test=None,
    y_test=None,
    results_dir: str = None,
    save_metrics: bool = True,
    save_plot: bool = True,
):
    """
    Evaluate model on test set; save metrics and confusion matrix.

    If model/X_test/y_test are None, loads from data.load_dataset and results/model.joblib.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    utils.ensure_dir(results_dir)

    if model is None or X_test is None or y_test is None:
        # Load model and data
        model_path = os.path.join(results_dir, MODEL_FILE)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"No saved model at {model_path}. Run train.py first."
            )
        payload = joblib.load(model_path)
        model = payload["model"]
        _, X_test, _, y_test, _, _ = data.load_dataset()

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if save_metrics:
        metrics_path = os.path.join(results_dir, METRICS_FILE)
        utils.save_json(metrics, metrics_path)
        print(f"Metrics saved to {metrics_path}")

    cm = confusion_matrix(y_test, y_pred)
    if save_plot:
        plot_path = os.path.join(results_dir, CONFUSION_PLOT)
        _plot_confusion_matrix(cm, plot_path)
        print(f"Confusion matrix saved to {plot_path}")

    return metrics, cm


def _plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    """Save confusion matrix heatmap to path."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if _USE_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Phishing"],
            yticklabels=["Benign", "Phishing"],
            ax=ax,
            cbar_kws={"label": "Count"},
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Phishing"])
        ax.set_yticklabels(["Benign", "Phishing"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate phishing detection model")
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory for metrics.json and confusion_matrix.png",
    )
    args = parser.parse_args()

    metrics, _ = evaluate(results_dir=args.results_dir)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
