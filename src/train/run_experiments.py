"""CardioSentinel experiment runner.

Stages
------
Stage 1 — baselines: LogReg, LogReg-balanced, Tree (XGB or HGBC).
Stage 2 — enhancements: interactions toggle, tree hyperparameter search.

All results are logged to MLflow and summarised in ``outputs/runs_summary.csv``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit

from src.data.load_data import load_heart_data
from src.eval.evaluation import (
    apply_threshold,
    compute_metrics,
    #find_threshold_at_precision,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_sweep,
)
from src.features.feature_spec import FactoryConfig, FeatureSpec, default_feature_spec
from src.pipeline.pipeline_factory import make_pipeline
from src.utils.mlflow_utils import log_plots, setup_experiment
from src.utils.seed import SEED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def _get_tree_estimator(random_state: int = SEED):
    XGB = _try_import_xgboost()
    if XGB is not None:
        return (
            "xgboost",
            XGB(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return (
        "hist_gbc",
        HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
        ),
    )


def _split_data(df: pd.DataFrame, target: str = "heart_attack_risk"):
    """Stratified 70/15/15 split. Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    y = df[target]
    X = df.drop(columns=[target])

    # First split: 70% train, 30% temp
    ss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(ss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    # Second split: 50/50 of temp -> 15%/15% of total
    ss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(ss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------
# Single-run evaluator
# ------------------------------------------------------------------

def _run_single(
    run_name: str,
    stage: str,
    model_type: str,
    cfg_name: str,
    estimator,
    feature_spec: FeatureSpec,
    cfg: FactoryConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    extra_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit pipeline, evaluate on test, log everything to MLflow."""
    pipe = make_pipeline(model_type, estimator, feature_spec, cfg)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "stage": stage,
            "model_type": model_type,
            "cfg_name": cfg_name,
        })
        mlflow.log_params({
            "model_name": run_name,
            "model_type": model_type,
            "cfg_name": cfg_name,
            "seed": SEED,
            "split_train": 0.70,
            "split_val": 0.15,
            "split_test": 0.15,
            "use_interactions_linear": cfg.use_interactions_linear,
            "use_country_risk_index": cfg.use_country_risk_index,
            "country_smoothing": cfg.country_smoothing,
            "drop_country_after_encoding": cfg.drop_country_after_encoding,
        })
        if extra_params:
            mlflow.log_params(extra_params)

        pipe.fit(X_train, y_train)

        # Probabilities & predictions
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred_default = pipe.predict(X_test)

        # Standard metrics at default 0.5 threshold
        std_metrics = compute_metrics(np.array(y_test), y_pred_default, y_proba, precision_floors=(0.30, 0.40),)

        all_metrics = std_metrics
        mlflow.log_metrics(all_metrics)

        # Plots
        threshold_30 = all_metrics["threshold_at_precision_30"]
        y_pred_30 = apply_threshold(y_proba, threshold_30)

        threshold_40 = all_metrics["threshold_at_precision_40"]
        y_pred_40 = apply_threshold(y_proba, threshold_40)

        figs = {
            "roc_curve": plot_roc_curve(np.array(y_test), y_proba),
            "pr_curve": plot_pr_curve(np.array(y_test), y_proba),
            "confusion_matrix_p30": plot_confusion_matrix(
                np.array(y_test),
                y_pred_30,
                title=f"CM @ precision≥0.30 (thr={threshold_30:.3f})",
            ),
            "confusion_matrix_p40": plot_confusion_matrix(
                np.array(y_test),
                y_pred_40,
                title=f"CM @ precision≥0.40 (thr={threshold_40:.3f})",
            ),
            "threshold_sweep": plot_threshold_sweep(np.array(y_test), y_proba),
        }
        log_plots(figs)

        logger.info(
            "%s | recall@p30=%.3f  recall@p40=%.3f  pr_auc=%.3f  roc_auc=%.3f",
            run_name,
            all_metrics.get("recall_at_precision_30", 0),
            all_metrics.get("recall_at_precision_40", 0),
            all_metrics.get("pr_auc", 0),
            all_metrics.get("roc_auc", 0),
        )

    return {
        "model_family": model_type,
        "cfg_name": cfg_name,
        "run_name": run_name,
        **all_metrics,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info("Loading data …")
    df = load_heart_data()
    logger.info("Dataset shape: %s  Target distribution:\n%s",
                df.shape, df["heart_attack_risk"].value_counts().to_string())

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(df)
    logger.info("Split sizes — train: %d  val: %d  test: %d",
                len(X_train), len(X_val), len(X_test))

    setup_experiment("CardioSentinel_Experiments")
    feature_spec = default_feature_spec()
    results: List[Dict[str, Any]] = []

    # ---- Stage 1 -------------------------------------------------------
    logger.info("=== Stage 1: baselines ===")

    # 1a) LogReg baseline
    cfg_base = FactoryConfig(use_interactions_linear=False)
    results.append(_run_single(
        run_name="logreg_baseline",
        stage="stage1", model_type="linear", cfg_name="baseline",
        estimator=LogisticRegression(solver="liblinear", max_iter=1000,
                                     random_state=SEED),
        feature_spec=feature_spec, cfg=cfg_base,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # 1b) LogReg balanced
    results.append(_run_single(
        run_name="logreg_balanced",
        stage="stage1", model_type="linear", cfg_name="balanced",
        estimator=LogisticRegression(solver="liblinear", max_iter=1000,
                                     class_weight="balanced",
                                     random_state=SEED),
        feature_spec=feature_spec, cfg=cfg_base,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # 1c) Tree baseline
    tree_name, tree_est = _get_tree_estimator()
    cfg_tree = FactoryConfig(use_interactions_tree=False)
    results.append(_run_single(
        run_name=f"tree_{tree_name}_baseline",
        stage="stage1", model_type="tree", cfg_name="tree_baseline",
        estimator=tree_est,
        feature_spec=feature_spec, cfg=cfg_tree,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # ---- Stage 2 -------------------------------------------------------
    logger.info("=== Stage 2: enhancements ===")

    # 2a) LogReg with interactions ON
    cfg_inter_on = FactoryConfig(use_interactions_linear=True)
    results.append(_run_single(
        run_name="logreg_interactions_on",
        stage="stage2", model_type="linear", cfg_name="interactions_on",
        estimator=LogisticRegression(solver="liblinear", max_iter=2000,
                                     random_state=SEED),
        feature_spec=feature_spec, cfg=cfg_inter_on,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # 2b) LogReg with interactions OFF (explicit comparison)
    cfg_inter_off = FactoryConfig(use_interactions_linear=False)
    results.append(_run_single(
        run_name="logreg_interactions_off",
        stage="stage2", model_type="linear", cfg_name="interactions_off",
        estimator=LogisticRegression(solver="liblinear", max_iter=2000,
                                     random_state=SEED),
        feature_spec=feature_spec, cfg=cfg_inter_off,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # 2c) Tree tuned via RandomizedSearchCV
    logger.info("Tuning tree model …")
    tree_name_t, tree_est_t = _get_tree_estimator()
    cfg_tree_tuned = FactoryConfig(use_interactions_tree=False)
    pipe_for_search = make_pipeline("tree", tree_est_t, feature_spec, cfg_tree_tuned)

    # Param grid — prefix with step name
    if tree_name_t == "xgboost":
        param_dist = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 1.0],
        }
    else:
        param_dist = {
            "model__max_iter": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 9, None],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__min_samples_leaf": [10, 20, 30],
        }

    search = RandomizedSearchCV(
        pipe_for_search,
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",
        cv=3,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    logger.info("Best tree params: %s", search.best_params_)

    # Evaluate best tuned tree on test set
    y_proba_tuned = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_tuned = best_pipe.predict(X_test)

    all_m = compute_metrics(
        np.array(y_test),
        y_pred_tuned,
        y_proba_tuned,
        precision_floors=(0.30, 0.40),
    )

    thr30 = all_m["threshold_at_precision_30"]
    y_pred_30 = apply_threshold(y_proba_tuned, thr30)

    thr40 = all_m["threshold_at_precision_40"]
    y_pred_40 = apply_threshold(y_proba_tuned, thr40)

    with mlflow.start_run(run_name=f"tree_{tree_name_t}_tuned"):
        mlflow.set_tags({"stage": "stage2", "model_type": "tree",
                         "cfg_name": "tree_tuned"})
        mlflow.log_params({
            "model_name": f"tree_{tree_name_t}_tuned",
            "model_type": "tree",
            "cfg_name": "tree_tuned",
            "seed": SEED,
            "split_train": 0.70, "split_val": 0.15, "split_test": 0.15,
            "best_params": str(search.best_params_),
            "search_n_iter": 30,
            "search_cv": 3,
        })
        mlflow.log_metrics(all_m)
        figs = {
            "roc_curve": plot_roc_curve(np.array(y_test), y_proba_tuned),
            "pr_curve": plot_pr_curve(np.array(y_test), y_proba_tuned),
            "confusion_matrix_p30": plot_confusion_matrix(
                np.array(y_test), y_pred_30,
                title=f"CM @ precision≥0.30 (thr={thr30:.3f})",
            ),
            "confusion_matrix_p40": plot_confusion_matrix(
                np.array(y_test), y_pred_40,
                title=f"CM @ precision≥0.40 (thr={thr40:.3f})",
            ),
            "threshold_sweep": plot_threshold_sweep(np.array(y_test), y_proba_tuned),
        }
        log_plots(figs)

    results.append({
        "model_family": "tree",
        "cfg_name": "tree_tuned",
        "run_name": f"tree_{tree_name_t}_tuned",
        **all_m,
    })

    # ---- Summary --------------------------------------------------------
    OUTPUTS_DIR.mkdir(exist_ok=True)
    summary = pd.DataFrame(results).fillna(np.nan).sort_values(
        by=["recall_at_precision_40", "pr_auc"],
        ascending=[False, False],
    )
    summary_path = OUTPUTS_DIR / "runs_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
