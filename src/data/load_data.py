"""Data loading with fallback to DATASET_PATH env var."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_heart_data() -> pd.DataFrame:
    """Load heart.csv and return a DataFrame with target cast to int.

    Resolution order:
      1. data/heart.csv relative to project root
      2. DATASET_PATH environment variable
    Raises FileNotFoundError with guidance if neither is available.
    """
    project_root = Path(__file__).resolve().parents[2]
    primary = project_root / "data" / "heart.csv"

    if primary.is_file():
        path = primary
    else:
        env_path = os.environ.get("DATASET_PATH")
        if env_path and Path(env_path).is_file():
            path = Path(env_path)
        else:
            raise FileNotFoundError(
                "heart.csv not found. Place it at data/heart.csv or set the "
                "DATASET_PATH environment variable to its absolute path."
            )

    df = pd.read_csv(path)

    # Rename target to snake_case if needed and cast to int
    target_col = "Heart Attack Risk"
    target_snake = "heart_attack_risk"
    if target_col in df.columns:
        df = df.rename(columns={target_col: target_snake})
    if target_snake not in df.columns:
        raise ValueError(f"Expected target column '{target_snake}' not found in data.")

    df[target_snake] = df[target_snake].astype(int)

    # Drop Patient ID — not a useful feature
    if "Patient ID" in df.columns:
        df = df.drop(columns=["Patient ID"])

    # Normalize column names to snake_case for consistency
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df
