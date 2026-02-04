"""CardioSentinel Phase 2 — Final model selection and packaging.

Retrains the chosen model (logreg_interactions_on) on train+val,
evaluates once on the held-out test set with a fixed threshold,
and writes deployment artifacts to ``artifacts/``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.load_data import load_heart_data
from src.eval.evaluation import (
    apply_threshold,
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.features.feature_spec import FactoryConfig, default_feature_spec
from src.pipeline.pipeline_factory import make_pipeline
from src.utils.mlflow_utils import log_plots, setup_experiment
from src.utils.seed import SEED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---- Locked decisions (Phase 2) ------------------------------------
FIXED_THRESHOLD = 0.380669721
PRECISION_FLOOR = 0.40
MODEL_TYPE = "linear"
CFG_NAME = "interactions_on"
RUN_NAME = "logreg_interactions_on"


# ------------------------------------------------------------------
# Data split — identical to Phase 1
# ------------------------------------------------------------------

def _split_data(df: pd.DataFrame, target: str = "heart_attack_risk"):
    """Stratified 70/15/15 split (mirrors Phase 1 exactly)."""
    y = df[target]
    X = df.drop(columns=[target])

    ss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(ss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    ss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(ss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1. Load data & split (identical to Phase 1)
    logger.info("Loading data...")
    df = load_heart_data()
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(df)

    # 2. Combine train + val
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    logger.info(
        "Train+val rows: %d  |  positive rate: %.4f",
        len(y_trainval),
        y_trainval.mean(),
    )

    # 3. Build pipeline (exact Phase 1 config for logreg_interactions_on)
    feature_spec = default_feature_spec()
    cfg = FactoryConfig(use_interactions_linear=True)
    estimator = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    pipe = make_pipeline(MODEL_TYPE, estimator, feature_spec, cfg)

    # 4. Fit on train+val
    logger.info("Fitting pipeline on train+val...")
    pipe.fit(X_trainval, y_trainval)

    # 5. Evaluate on test set
    logger.info("Evaluating on held-out test set...")
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred_fixed = apply_threshold(y_proba, FIXED_THRESHOLD)

    metrics = compute_metrics(
        y_true=y_test.values,
        y_pred=y_pred_fixed,
        y_proba=y_proba,
        precision_floors=(0.40,),
    )

    test_metrics = {
        "recall_at_precision_40": metrics["recall_at_precision_40"],
        "precision_at_precision_40": metrics["precision_at_precision_40"],
        "predicted_positive_rate": float((y_pred_fixed == 1).mean()),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }

    for k, v in test_metrics.items():
        logger.info("  %s = %.6f", k, v)

    # 6. Generate plots
    fig_roc = plot_roc_curve(y_test.values, y_proba)
    fig_pr = plot_pr_curve(y_test.values, y_proba)
    fig_cm = plot_confusion_matrix(
        y_test.values, y_pred_fixed, title="Confusion Matrix (p≥0.40 threshold)"
    )

    # 7. Save artifacts to disk
    logger.info("Saving artifacts to %s", ARTIFACTS_DIR)

    # model.joblib
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(pipe, model_path)

    # plots
    fig_roc.savefig(ARTIFACTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    fig_pr.savefig(ARTIFACTS_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    fig_cm.savefig(
        ARTIFACTS_DIR / "confusion_matrix_p40.png", dpi=150, bbox_inches="tight"
    )

    # model_card.json
    model_card = {
        "model_name": RUN_NAME,
        "model_type": "logistic_regression",
        "selection_metric": "recall_at_precision_40",
        "precision_floor": PRECISION_FLOOR,
        "threshold": FIXED_THRESHOLD,
        "test_metrics": test_metrics,
        "training_data": {
            "rows": int(len(y_trainval)),
            "positive_rate": float(y_trainval.mean()),
        },
        "intended_use": "Screening support, not diagnosis",
        "limitations": [
            "Performance depends on threshold choice",
            "Not calibrated for absolute risk probabilities",
            "Trained on a single dataset",
        ],
        "ethical_notes": [
            "False positives may cause unnecessary concern",
            "False negatives may miss at-risk individuals",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    card_path = ARTIFACTS_DIR / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    logger.info("Model card written to %s", card_path)

    # 8. Log to MLflow
    logger.info("Logging to MLflow...")
    setup_experiment("CardioSentinel_Experiments")

    with mlflow.start_run(run_name=f"{RUN_NAME}_final"):
        mlflow.set_tags(
            {
                "stage": "stage3",
                "final_model": "true",
                "model_type": MODEL_TYPE,
            }
        )
        mlflow.log_params(
            {
                "model_name": RUN_NAME,
                "model_type": MODEL_TYPE,
                "cfg_name": CFG_NAME,
                "seed": SEED,
                "threshold": FIXED_THRESHOLD,
                "precision_floor": PRECISION_FLOOR,
                "train_val_rows": len(y_trainval),
                "test_rows": len(y_test),
                "use_interactions_linear": cfg.use_interactions_linear,
                "use_country_risk_index": cfg.use_country_risk_index,
                "country_smoothing": cfg.country_smoothing,
                "drop_country_after_encoding": cfg.drop_country_after_encoding,
            }
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)
        # Also log the standard metrics produced by compute_metrics
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            if k in metrics and k not in test_metrics:
                mlflow.log_metric(k, metrics[k])

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(card_path))
        log_plots(
            {
                "roc_curve": fig_roc,
                "pr_curve": fig_pr,
                "confusion_matrix_p40": fig_cm,
            }
        )

    logger.info("Phase 2 complete. Artifacts saved to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
