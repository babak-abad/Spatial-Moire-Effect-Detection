from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_NAMES = ("clean", "moire")


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(dtype=np.int64)
    matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "threshold": threshold,
        "num_samples": int(labels.size),
        "accuracy": float(accuracy_score(y_true=labels, y_pred=predictions)),
        "precision": float(precision_score(y_true=labels, y_pred=predictions, zero_division=0)),
        "recall": float(recall_score(y_true=labels, y_pred=predictions, zero_division=0)),
        "f1": float(f1_score(y_true=labels, y_pred=predictions, zero_division=0)),
        "confusion_matrix": {
            "rows": "actual (clean, moire)",
            "columns": "predicted (clean, moire)",
            "matrix": [[tn, fp], [fn, tp]],
        },
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if np.unique(ar=labels).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true=labels, y_score=probabilities))
    return metrics


def classification_report_text(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> str:
    predictions = (probabilities >= threshold).astype(dtype=np.int64)
    return classification_report(
        y_true=labels,
        y_pred=predictions,
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )


def plot_confusion_matrix(labels: np.ndarray, probabilities: np.ndarray, output_path: Path, threshold: float = 0.5) -> None:
    predictions = (probabilities >= threshold).astype(dtype=np.int64)
    matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(X=matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(mappable=image, ax=axis)
    axis.set_xticks(ticks=[0, 1])
    axis.set_xticklabels(labels=CLASS_NAMES)
    axis.set_yticks(ticks=[0, 1])
    axis.set_yticklabels(labels=CLASS_NAMES)
    axis.set_xlabel(xlabel="predicted label")
    axis.set_ylabel(ylabel="true label")
    axis.set_title(label=f"confusion matrix (threshold = {threshold:.2f})")
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(x=col, y=row, s=str(value), ha="center", va="center", color=color, fontsize=13)
    figure.savefig(fname=output_path, dpi=150, bbox_inches="tight")
    plt.close(fig=figure)


def plot_training_history(history: dict, output_path: Path) -> None:
    epochs = history["epoch"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (loss_axis, score_axis) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))
    loss_axis.plot(epochs, history["train_loss"], marker="o", label="train loss")
    loss_axis.plot(epochs, history["val_loss"], marker="s", label="val loss")
    loss_axis.set_xlabel(xlabel="epoch")
    loss_axis.set_ylabel(ylabel="BCE loss")
    loss_axis.set_title(label="loss")
    loss_axis.legend()
    loss_axis.grid(visible=True, alpha=0.3)
    score_axis.plot(epochs, history["train_accuracy"], marker="o", label="train accuracy")
    score_axis.plot(epochs, history["val_accuracy"], marker="s", label="val accuracy")
    score_axis.plot(epochs, history["val_f1"], marker="^", label="val F1")
    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        score_axis.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label="best epoch")
    score_axis.set_xlabel(xlabel="epoch")
    score_axis.set_ylabel(ylabel="score")
    score_axis.set_ylim(bottom=0.0, top=1.05)
    score_axis.set_title(label="accuracy / F1")
    score_axis.legend(loc="lower right")
    score_axis.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(fname=output_path, dpi=150)
    plt.close(fig=figure)


def save_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=json.dumps(metrics, indent=2), encoding="utf-8")
