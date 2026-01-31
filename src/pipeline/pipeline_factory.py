"""Pipeline factory — assembles full sklearn Pipeline with BP split,
preprocessing, optional interactions, and the estimator.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.features.feature_spec import FactoryConfig, FeatureSpec
from src.pipeline.preprocessor import BloodPressureSplitter, build_preprocessor


class _NumericInteractionAdder:
    """Post-preprocessor step that applies PolynomialFeatures only to the
    first *n_numeric* columns of the preprocessor output (those correspond
    to the numeric pipeline) and concatenates back the rest.

    This avoids expanding the one-hot / binary columns with interaction
    terms.
    """

    def __init__(self, n_numeric: int):
        self.n_numeric = n_numeric
        self.poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )

    def fit(self, X, y=None):  # noqa: N803
        self.poly.fit(X[:, : self.n_numeric])
        return self

    def transform(self, X):  # noqa: N803
        num_part = self.poly.transform(X[:, : self.n_numeric])
        rest = X[:, self.n_numeric :]
        return np.hstack([num_part, rest])

    def fit_transform(self, X, y=None):  # noqa: N803
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"n_numeric": self.n_numeric}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def make_pipeline(
    model_type: str,
    estimator,
    feature_spec: FeatureSpec,
    cfg: FactoryConfig,
) -> Pipeline:
    """Build a complete sklearn Pipeline.

    Steps
    -----
    1. ``bp_split`` — :class:`BloodPressureSplitter`
    2. ``preprocess`` — :class:`ColumnTransformer`
    3. (optional) ``interactions`` — polynomial interaction on numeric cols
    4. ``model`` — the supplied estimator

    Parameters
    ----------
    model_type : ``"linear"`` or ``"tree"``
    estimator : scikit-learn compatible classifier
    feature_spec : column lists
    cfg : factory knobs
    """
    preprocessor = build_preprocessor(model_type, feature_spec, cfg)

    steps: list = [
        ("bp_split", BloodPressureSplitter()),
        ("preprocess", preprocessor),
    ]

    # Add interactions for linear models if requested
    want_interactions = (
        (model_type == "linear" and cfg.use_interactions_linear)
        or (model_type == "tree" and cfg.use_interactions_tree)
    )
    if want_interactions:
        n_numeric = len(feature_spec.numeric)
        steps.append(("interactions", _NumericInteractionAdder(n_numeric)))

    steps.append(("model", estimator))
    return Pipeline(steps)
