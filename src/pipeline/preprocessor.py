"""Custom transformers and preprocessor builder.

BloodPressureSplitter — deterministic / safe transform.
CountryRiskTransformer — learned target-encoding with Bayesian smoothing.
build_preprocessor — assembles a ColumnTransformer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_spec import FactoryConfig, FeatureSpec


# ---------------------------------------------------------------------------
# A) BloodPressureSplitter — safe / deterministic
# ---------------------------------------------------------------------------

class BloodPressureSplitter(BaseEstimator, TransformerMixin):
    """Split a ``blood_pressure`` string column (e.g. '120/80') into two
    numeric columns: ``systolic_bp`` and ``diastolic_bp``.

    Invalid or missing values produce ``np.nan``; downstream imputers
    handle them.  The original ``blood_pressure`` column is dropped from
    the output so it is not passed to the ColumnTransformer.
    """

    def __init__(self, bp_column: str = "blood_pressure"):
        self.bp_column = bp_column

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if self.bp_column in X.columns:
            systolic = []
            diastolic = []
            for val in X[self.bp_column]:
                try:
                    parts = str(val).strip().split("/")
                    systolic.append(float(parts[0]))
                    diastolic.append(float(parts[1]))
                except (ValueError, IndexError, TypeError):
                    systolic.append(np.nan)
                    diastolic.append(np.nan)
            X["systolic_bp"] = systolic
            X["diastolic_bp"] = diastolic
            X = X.drop(columns=[self.bp_column])
        return X


# ---------------------------------------------------------------------------
# B) CountryRiskTransformer — learned / Bayesian smoothing
# ---------------------------------------------------------------------------

class CountryRiskTransformer(BaseEstimator, TransformerMixin):
    """Target-encode ``country`` with Bayesian smoothing.

    Parameters
    ----------
    smoothing : float
        Smoothing factor *m*.  Higher values shrink rare-country estimates
        toward the global mean.
    country_column : str
        Name of the input column.
    output_column : str
        Name of the output column.
    """

    def __init__(
        self,
        smoothing: float = 20.0,
        country_column: str = "country",
        output_column: str = "country_risk_index",
    ):
        self.smoothing = smoothing
        self.country_column = country_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        if y is None:
            raise ValueError("CountryRiskTransformer requires y during fit.")
        col = X[self.country_column]
        y_arr = np.asarray(y)
        self.global_mean_: float = float(y_arr.mean())
        self.mapping_: dict[str, float] = {}

        for country in col.unique():
            mask = col == country
            n = int(mask.sum())
            mu_country = float(y_arr[mask].mean())
            w = n / (n + self.smoothing)
            self.mapping_[country] = w * mu_country + (1 - w) * self.global_mean_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        X[self.output_column] = (
            X[self.country_column]
            .map(self.mapping_)
            .fillna(self.global_mean_)
        )
        return X[[self.output_column]]


# ---------------------------------------------------------------------------
# C) build_preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(
    model_type: str,
    feature_spec: FeatureSpec,
    cfg: FactoryConfig,
) -> ColumnTransformer:
    """Build a :class:`ColumnTransformer` that handles numeric, binary,
    categorical, and (optionally) country-risk encoding.

    Parameters
    ----------
    model_type : ``"linear"`` or ``"tree"``
    feature_spec : column lists
    cfg : factory knobs
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    drop_value: Optional[str] = (
        cfg.onehot_drop_linear if model_type == "linear" else cfg.onehot_drop_tree
    )
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=drop_value,
                                 sparse_output=False)),
    ])

    # Decide which categorical columns go to one-hot encoding
    cat_cols = list(feature_spec.categorical)
    if cfg.use_country_risk_index and "country" in cat_cols and cfg.drop_country_after_encoding:
        cat_cols = [c for c in cat_cols if c != "country"]

    transformers: list = [
        ("num", numeric_pipe, feature_spec.numeric),
        ("bin", binary_pipe, feature_spec.binary),
        ("cat", cat_pipe, cat_cols),
    ]

    if cfg.use_country_risk_index and "country" in feature_spec.categorical:
        country_pipe = CountryRiskTransformer(
            smoothing=cfg.country_smoothing,
            country_column="country",
            output_column="country_risk_index",
        )
        transformers.append(("country_risk", country_pipe, ["country"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")
