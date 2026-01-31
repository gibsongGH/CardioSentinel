"""Feature specification and factory configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureSpec:
    """Lists of feature column names by type.

    ``numeric`` includes raw numeric columns plus engineered ones
    (systolic_bp, diastolic_bp) produced by BloodPressureSplitter.
    ``binary`` includes 0/1 indicator columns.
    ``categorical`` includes unordered categorical string columns.
    """

    numeric: List[str] = field(default_factory=list)
    binary: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)


@dataclass
class FactoryConfig:
    """Knobs that control how the preprocessing pipeline is built."""

    onehot_drop_linear: Optional[str] = "first"
    onehot_drop_tree: Optional[str] = None
    use_interactions_linear: bool = True
    use_interactions_tree: bool = False
    use_country_risk_index: bool = True
    country_smoothing: float = 20.0
    drop_country_after_encoding: bool = True


def default_feature_spec() -> FeatureSpec:
    """Return the default FeatureSpec for the heart attack dataset.

    Assumes BloodPressureSplitter has already added systolic_bp /
    diastolic_bp and that blood_pressure (raw string) is consumed by
    the splitter and NOT passed downstream.

    Column assignment (after snake_case normalisation):
      - Numeric continuous: age, cholesterol, heart_rate,
        exercise_hours_per_week, stress_level, sedentary_hours_per_day,
        income, bmi, triglycerides, physical_activity_days_per_week,
        sleep_hours_per_day, systolic_bp, diastolic_bp
      - Binary 0/1: diabetes, family_history, smoking, obesity,
        alcohol_consumption, previous_heart_problems, medication_use
      - Categorical (string): sex, diet, country, continent, hemisphere
    """
    return FeatureSpec(
        numeric=[
            "age",
            "cholesterol",
            "heart_rate",
            "exercise_hours_per_week",
            "stress_level",
            "sedentary_hours_per_day",
            "income",
            "bmi",
            "triglycerides",
            "physical_activity_days_per_week",
            "sleep_hours_per_day",
            "systolic_bp",
            "diastolic_bp",
        ],
        binary=[
            "diabetes",
            "family_history",
            "smoking",
            "obesity",
            "alcohol_consumption",
            "previous_heart_problems",
            "medication_use",
        ],
        categorical=[
            "sex",
            "diet",
            "country",
            "continent",
            "hemisphere",
        ],
    )
