"""Shared evaluation utilities: one place where every model is scored and logged.

Metrics
-------
* accuracy: dominated by the two big classes (politics, sport).
* macro-F1: the unweighted mean F1 over the 10 classes. This is our **primary metric**,
  because with a 24:1 imbalance a model can reach high accuracy while ignoring the
  small classes.
* weighted-F1 and macro precision / recall are reported for completeness.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from src import plotstyle
from src.data import RESULTS_DIR

EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
FIELDS = ["timestamp", "family", "model", "config", "seed", "split",
          "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "train_seconds"]


def compute_metrics(y_true, y_pred) -> dict:
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "macro_precision": round(p, 4),
        "macro_recall": round(r, 4),
    }


def log_experiment(family: str, model: str, config: dict | str, split: str, metrics: dict,
                   seed: int = 42, train_seconds: float | None = None) -> None:
    """Append one row to results/experiments.csv (creates the header if needed)."""
    new = not EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "family": family, "model": model,
            "config": config if isinstance(config, str) else json.dumps(config, sort_keys=True),
            "seed": seed, "split": split,
            "train_seconds": None if train_seconds is None else round(train_seconds, 1),
            **{k: metrics[k] for k in FIELDS if k in metrics},
        })


def save_report(name: str, y_true, y_pred, labels: list[str]) -> dict:
    """Write the per-class report (json) and confusion-matrix figure for one model."""
    out_dir = RESULTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(rep, f, indent=2)
    plot_confusion(y_true, y_pred, labels, RESULTS_DIR / "figures" / f"cm_{name}.png", title=name)
    return rep


def plot_confusion(y_true, y_pred, labels: list[str], path: Path, title: str) -> None:
    """Row-normalised confusion matrix (each row = true class, sums to 100%), with raw counts."""
    plotstyle.apply()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap=plotstyle.SEQ_CMAP, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else plotstyle.INK)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"Confusion matrix: {title}", loc="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
