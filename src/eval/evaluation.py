"""Evaluation metrics, custom thresholding, and plot generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

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
# Helpers
# ------------------------------------------------------------------

def _as_1d_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    return y.astype(int)


def _as_1d_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    return y.astype(float)


def _validate_binary(y: np.ndarray) -> None:
    vals = set(np.unique(y).tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"y must be binary 0/1. Found values: {sorted(vals)}")


@dataclass(frozen=True)
class PrecisionFloorResult:
    precision_floor: float
    threshold: float
    precision: float
    recall: float
    predicted_positive_rate: float


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions at *threshold*."""
    y_proba = _as_1d_float(y_proba)
    return (y_proba >= float(threshold)).astype(int)


# ------------------------------------------------------------------
# Custom thresholding at precision floor
# ------------------------------------------------------------------

def find_threshold_at_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    precision_floor: float = 0.30,
) -> Tuple[float, float, float]:
    """Find the lowest decision threshold that achieves *precision_floor*.

    Uses precision_recall_curve and chooses the *lowest* threshold where
    precision >= precision_floor (which tends to maximize recall under that constraint).

    Returns:
        (threshold, recall_at_threshold, precision_at_threshold)

    If no threshold achieves the precision floor, returns:
        threshold=1.0, recall=0.0, precision=1.0 (predict none positive)
    """
    y_true = _as_1d_int(y_true)
    y_proba = _as_1d_float(y_proba)
    _validate_binary(y_true)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precisions and recalls have length len(thresholds) + 1
    if thresholds.size == 0:
        logger.warning(
            "No thresholds produced by precision_recall_curve; returning threshold=1.0"
        )
        return 1.0, 0.0, 1.0

    # Align arrays to threshold indices: thresholds corresponds to precisions[:-1], recalls[:-1]
    p_t = precisions[:-1]
    r_t = recalls[:-1]
    t = thresholds

    qualifying = np.where(p_t >= float(precision_floor))[0]
    if qualifying.size == 0:
        logger.warning(
            "No threshold achieves precision >= %.2f; returning threshold=1.0",
            precision_floor,
        )
        return 1.0, 0.0, 1.0

    # Choose the index with the *lowest* threshold among qualifying (maximizes recall)
    best_idx = qualifying[np.argmin(t[qualifying])]

    return float(t[best_idx]), float(r_t[best_idx]), float(p_t[best_idx])


def precision_floor_results(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    precision_floors: Iterable[float] = (0.30,),
) -> Dict[float, PrecisionFloorResult]:
    """Compute threshold/precision/recall/pred_pos_rate for multiple precision floors."""
    y_true = _as_1d_int(y_true)
    y_proba = _as_1d_float(y_proba)
    _validate_binary(y_true)

    results: Dict[float, PrecisionFloorResult] = {}
    n = float(len(y_true))

    for floor in precision_floors:
        thr, rec, prec = find_threshold_at_precision(y_true, y_proba, precision_floor=float(floor))
        y_hat = apply_threshold(y_proba, thr)
        pred_pos_rate = float(y_hat.mean()) if n > 0 else 0.0

        results[float(floor)] = PrecisionFloorResult(
            precision_floor=float(floor),
            threshold=float(thr),
            precision=float(prec),
            recall=float(rec),
            predicted_positive_rate=float(pred_pos_rate),
        )

    return results


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    precision_floors: Iterable[float] = (0.30,),
) -> Dict[str, float]:
    """Return a dict of standard classification metrics + optional precision-floor metrics.

    Standard metrics (always):
      - accuracy, precision, recall, f1

    If y_proba is provided:
      - roc_auc, pr_auc
      - For each precision floor p (e.g., 0.30, 0.40), logs:
          recall_at_precision_30
          threshold_at_precision_30
          precision_at_precision_30
          pred_pos_rate_at_precision_30
        (same pattern for 40, 50, etc.)
    """
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    _validate_binary(y_true)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        y_proba = _as_1d_float(y_proba)

        # AUCs can error if a split has one class (unlikely with stratify, but safe)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["pr_auc"] = float("nan")

        floor_results = precision_floor_results(y_true, y_proba, precision_floors=precision_floors)
        for floor, res in floor_results.items():
            pct = int(round(floor * 100))
            metrics[f"recall_at_precision_{pct}"] = float(res.recall)
            metrics[f"threshold_at_precision_{pct}"] = float(res.threshold)
            metrics[f"precision_at_precision_{pct}"] = float(res.precision)
            metrics[f"pred_pos_rate_at_precision_{pct}"] = float(res.predicted_positive_rate)

    return metrics


# ------------------------------------------------------------------
# Plots (matplotlib only)
# ------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    y_true = _as_1d_int(y_true)
    y_proba = _as_1d_float(y_proba)
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
    y_true = _as_1d_int(y_true)
    y_proba = _as_1d_float(y_proba)
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
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    y_true = _as_1d_int(y_true)
    y_proba = _as_1d_float(y_proba)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    if thresholds.size > 0:
        ax.plot(thresholds, precisions[:-1], linewidth=2, label="Precision")
        ax.plot(thresholds, recalls[:-1], linewidth=2, label="Recall")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision & Recall")
    ax.legend()
    fig.tight_layout()
    return fig
