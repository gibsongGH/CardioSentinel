"""Inference module for CardioSentinel.

Loads the Phase 2 model artifacts and exposes a single-row prediction function.
No user data is logged or persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd


# Raw columns the pipeline expects *before* any preprocessing.
# blood_pressure is consumed by BloodPressureSplitter; systolic_bp / diastolic_bp
# are engineered downstream and must NOT appear in user input.
# heart_attack_risk (target) is excluded.
# Patient ID was dropped during training data loading.
RAW_INPUT_COLUMNS = [
    "age",
    "sex",
    "cholesterol",
    "blood_pressure",
    "heart_rate",
    "diabetes",
    "family_history",
    "smoking",
    "obesity",
    "alcohol_consumption",
    "exercise_hours_per_week",
    "diet",
    "previous_heart_problems",
    "medication_use",
    "stress_level",
    "sedentary_hours_per_day",
    "income",
    "bmi",
    "triglycerides",
    "physical_activity_days_per_week",
    "sleep_hours_per_day",
    "country",
    "continent",
    "hemisphere",
]

_REQUIRED_CARD_KEYS = {"threshold", "precision_floor", "model_name"}


# Columns expected to be numeric in the training pipeline (coerce invalid -> NaN).
_NUMERIC_COLS = [
    "age",
    "cholesterol",
    "heart_rate",
    "alcohol_consumption",
    "exercise_hours_per_week",
    "stress_level",
    "sedentary_hours_per_day",
    "income",
    "bmi",
    "triglycerides",
    "physical_activity_days_per_week",
    "sleep_hours_per_day",
]

# Columns expected to be binary-ish 0/1 in the training pipeline (coerce common forms -> 0/1).
_BINARY_COLS = [
    "diabetes",
    "family_history",
    "smoking",
    "obesity",
    "previous_heart_problems",
    "medication_use",
]

# Columns treated as categorical strings.
_CATEGORICAL_COLS = ["sex", "diet", "country", "continent", "hemisphere"]


def load_model(artifacts_dir: str | Path = "artifacts") -> Tuple[Any, Dict[str, Any]]:
    """Load the serialized pipeline and model card from *artifacts_dir*.

    Returns
    -------
    pipeline : sklearn Pipeline
    model_card : dict
    """
    artifacts_dir = Path(artifacts_dir)

    model_path = artifacts_dir / "model.joblib"
    card_path = artifacts_dir / "model_card.json"

    if not model_path.is_file() or not card_path.is_file():
        raise FileNotFoundError(
            f"Model artifacts not found in {artifacts_dir.resolve()}. "
            "Run Phase 2 first: python -m src.train.finalize_model"
        )

    pipeline = joblib.load(model_path)

    with open(card_path, "r", encoding="utf-8") as f:
        model_card: Dict[str, Any] = json.load(f)

    missing = _REQUIRED_CARD_KEYS - set(model_card)
    if missing:
        raise ValueError(
            f"model_card.json is missing required keys: {sorted(missing)}. "
            "Re-run Phase 2 to regenerate artifacts."
        )

    # Basic sanity checks
    try:
        _ = float(model_card["threshold"])
        _ = float(model_card["precision_floor"])
    except Exception as e:
        raise ValueError(
            "model_card.json has invalid numeric values for threshold/precision_floor."
        ) from e

    return pipeline, model_card


def _to01(x: Any) -> Any:
    """Coerce common truthy/falsey representations to 0/1; otherwise NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if np.isfinite(x):
            return int(x)
        return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return 1
        if s in {"0", "false", "f", "no", "n"}:
            return 0
    return np.nan


def _positive_class_index_from_pipeline(pipeline: Any) -> int:
    """Return the index into predict_proba() corresponding to class 1.

    Falls back to index 1 if class info is not available.
    """
    # Default fallback aligns with common sklearn behavior [class0, class1]
    pos_index = 1

    try:
        estimator = None

        # Preferred: pipeline named step "model"
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            estimator = pipeline.named_steps["model"]

        # Fallback: try last step if it's a Pipeline
        if estimator is None and hasattr(pipeline, "steps") and pipeline.steps:
            estimator = pipeline.steps[-1][1]

        classes = getattr(estimator, "classes_", None)
        if classes is None:
            return pos_index

        classes = np.array(classes)
        if 1 in classes:
            return int(np.where(classes == 1)[0][0])

        # If labels aren't {0,1}, still try the "positive" as max label
        # (documented fallback; keeps behavior deterministic)
        return int(np.argmax(classes))

    except Exception:
        return pos_index


def predict_one(
    input_dict: Dict[str, Any],
    pipeline: Any,
    model_card: Dict[str, Any],
) -> Dict[str, Any]:
    """Run inference on a single observation.

    Parameters
    ----------
    input_dict : dict
        Raw feature values keyed by snake_case column name. Must contain all
        columns listed in RAW_INPUT_COLUMNS.
    pipeline : sklearn Pipeline
        The fitted pipeline loaded by :func:`load_model`.
    model_card : dict
        The model card dict loaded by :func:`load_model`.

    Returns
    -------
    dict with keys: risk_score, is_high_risk, threshold_used,
    precision_floor, model_name.
    """
    missing = set(RAW_INPUT_COLUMNS) - set(input_dict)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    # Build a single-row dataframe in the expected order.
    row = pd.DataFrame([{col: input_dict.get(col) for col in RAW_INPUT_COLUMNS}])

    # ---- Type coercion (align Streamlit inputs with training expectations) ----
    # Normalize blood pressure string
    row["blood_pressure"] = row["blood_pressure"].astype(str).str.replace(" ", "", regex=False)

    # Coerce numeric cols -> float
    row[_NUMERIC_COLS] = row[_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").astype("float64")

    # Coerce binary cols -> float (0/1 with NaN allowed)
    for c in _BINARY_COLS:
        row[c] = row[c].map(_to01)
    row[_BINARY_COLS] = row[_BINARY_COLS].apply(pd.to_numeric, errors="coerce").astype("float64")

    # Categoricals -> plain Python strings (object)
    for c in _CATEGORICAL_COLS:
        row[c] = row[c].astype(object)

    # ---- Predict probability robustly (do not assume column 1 is positive) ----
    try:
        proba_vec = pipeline.predict_proba(row)[0]
    except Exception as e:
        # Helpful crash report that will pinpoint the bad column/type
        raise RuntimeError(
            "Prediction failed.\n"
            f"dtypes:\n{row.dtypes}\n\n"
            f"row:\n{row.iloc[0].to_dict()}\n"
        ) from e

    proba_vec = pipeline.predict_proba(row)[0]
    pos_idx = _positive_class_index_from_pipeline(pipeline)
    proba_pos = float(proba_vec[pos_idx])

    threshold = float(model_card["threshold"])

    return {
        "risk_score": proba_pos,
        "is_high_risk": bool(proba_pos >= threshold),
        "threshold_used": threshold,
        "precision_floor": float(model_card["precision_floor"]),
        "model_name": model_card["model_name"],
    }
