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


def load_model(
    artifacts_dir: str | Path = "artifacts",
) -> Tuple[Any, Dict[str, Any]]:
    """Load the serialised pipeline and model card from *artifacts_dir*.

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
            "Run Phase 2 first:  python -m src.train.finalize_model"
        )

    pipeline = joblib.load(model_path)

    with open(card_path, "r", encoding="utf-8") as f:
        model_card: Dict[str, Any] = json.load(f)

    missing = _REQUIRED_CARD_KEYS - set(model_card)
    if missing:
        raise ValueError(
            f"model_card.json is missing required keys: {missing}. "
            "Re-run Phase 2 to regenerate artifacts."
        )

    return pipeline, model_card


def predict_one(
    input_dict: Dict[str, Any],
    pipeline: Any,
    model_card: Dict[str, Any],
) -> Dict[str, Any]:
    """Run inference on a single observation.

    Parameters
    ----------
    input_dict : dict
        Raw feature values keyed by snake_case column name.  Must contain at
        least all columns listed in ``RAW_INPUT_COLUMNS``.
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

    row = pd.DataFrame([{col: input_dict[col] for col in RAW_INPUT_COLUMNS}])

    proba = pipeline.predict_proba(row)[:, 1][0]
    threshold = float(model_card["threshold"])

    return {
        "risk_score": float(proba),
        "is_high_risk": bool(proba >= threshold),
        "threshold_used": threshold,
        "precision_floor": float(model_card["precision_floor"]),
        "model_name": model_card["model_name"],
    }
