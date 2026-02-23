"""Utility functions for phishing detection pipeline."""
import os
import json


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save dict to JSON file."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
