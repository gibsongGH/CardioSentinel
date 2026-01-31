# CardioSentinel

Heart attack risk prediction pipeline with MLflow experiment tracking.

## What it does

CardioSentinel trains and compares multiple classifiers (Logistic Regression,
XGBoost / HistGradientBoosting) on a heart attack risk dataset.  Each
experiment is logged to MLflow with metrics, parameters, and diagnostic plots.

### Key metric: `recall_at_precision_30`

In clinical screening we want to **catch as many at-risk patients as possible**
(high recall) while keeping the false-positive rate tolerable.
`recall_at_precision_30` answers: *"If we lower the decision threshold until at
least 30 % of flagged patients truly are at risk (precision >= 0.30), what
fraction of all actual at-risk patients do we catch?"*  Higher is better.

## Quick start

```bash
# 1. Install dependencies (no seaborn; xgboost is optional)
pip install -r requirements.txt

# 2. Run all experiments
python -m src.train.run_experiments

# 3. Browse results in MLflow
mlflow ui --backend-store-uri file:./mlruns
# Then open http://127.0.0.1:5000 in your browser.
```

## Project structure

```
data/heart.csv              <- dataset (not committed)
outputs/runs_summary.csv    <- experiment comparison table
mlruns/                     <- MLflow tracking store
src/
  data/load_data.py         <- CSV loader with env-var fallback
  features/feature_spec.py  <- FeatureSpec + FactoryConfig dataclasses
  pipeline/preprocessor.py  <- BP splitter, country risk encoder, ColumnTransformer
  pipeline/pipeline_factory.py <- full sklearn Pipeline assembly
  eval/evaluation.py        <- metrics, threshold logic, matplotlib plots
  train/run_experiments.py  <- Stage 1 + Stage 2 experiment runner
  utils/seed.py             <- SEED = 42
  utils/mlflow_utils.py     <- MLflow setup & artifact helpers
```

## Data

Place `heart.csv` in `data/`, or set `DATASET_PATH=/abs/path/to/heart.csv`.

## Reproducibility

- Single seed constant: `SEED = 42` (src/utils/seed.py).
- Stratified 70 / 15 / 15 train / val / test split.
- Seed and split config are logged to every MLflow run.

## Phase 1 – Implementation Notes

- Project scaffold, preprocessing, evaluation, and MLflow experiments implemented via agent-driven rebuild.
- recall_at_precision_30 reaching 1.0 across models reflects dataset base rate (~35.8%) exceeding the 0.30 precision floor at low thresholds; this is expected.
- See outputs/runs_summary.csv for comparative results.