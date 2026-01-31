"""Evaluation metrics, custom thresholding, and plot generation."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return a dict of standard classification metrics."""
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


# ------------------------------------------------------------------
# Custom threshold at precision >= 0.30
# ------------------------------------------------------------------

def find_threshold_at_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    precision_floor: float = 0.30,
) -> Tuple[float, float]:
    """Find the lowest decision threshold that achieves *precision_floor*.

    Returns (threshold, recall_at_that_threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precisions and recalls have length len(thresholds) + 1.
    # Only indices up to len(thresholds)-1 have a corresponding threshold.
    qualifying = np.where(precisions[:-1] >= precision_floor)[0]
    if len(qualifying) == 0:
        logger.warning(
            "No threshold achieves precision >= %.2f; returning threshold=1.0",
            precision_floor,
        )
        return 1.0, 0.0

    # Among qualifying, choose the index with the *lowest* threshold
    # (which gives the highest recall).
    best_idx = qualifying[np.argmin(thresholds[qualifying])]
    return float(thresholds[best_idx]), float(recalls[best_idx])


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions at *threshold*."""
    return (y_proba >= threshold).astype(int)


# ------------------------------------------------------------------
# Plots (matplotlib only)
# ------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precisions[:-1], linewidth=2, label="Precision")
    ax.plot(thresholds, recalls[:-1], linewidth=2, label="Recall")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision & Recall")
    ax.legend()
    fig.tight_layout()
    return fig
