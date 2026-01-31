CardioSentinel - Revised Build Specification
Goal: Build a robust, end-to-end ML pipeline and deployment for heart attack risk prediction. Stack: Python 3.10+, Scikit-Learn, MLflow, Streamlit, Docker.

1. Project Setup
File Structure:

.
├── artifacts/          # Generated models/plots (gitignored)
├── data/               # Local data storage
│   └── heart.csv       # Sourced from user
├── outputs/            # Experiment summaries
├── mlruns/             # MLflow tracking store
├── src/
│   ├── data/           # Loading logic
│   ├── features/       # Feature specs & custom transformers
│   ├── pipeline/       # Pipeline factory
│   ├── train/          # Experiment runners
│   ├── eval/           # Metrics & plots
│   ├── infer/          # Inference logic
│   └── utils/          # Logging, seeding
├── app/
│   └── streamlit_app.py
├── Dockerfile
└── requirements.txt
Dependencies (requirements.txt):

pandas>=2.0.0
scikit-learn>=1.3.0
mlflow>=2.5.0
streamlit>=1.25.0
xgboost (optional, fallback to HistGradientBoostingClassifier)
matplotlib, seaborn
Utilities:

src/utils/seed.py: Define SEED = 42. Ensure this is passed to all splitters and estimators.
src/utils/mlflow_utils.py:
setup_experiment(name): Sets tracking URI to file://<abs_path>/mlruns.
log_plots(dict_of_figs): Helper to save matplotlib figures as artifacts.
2. Data & Features
Data Source:

Read from data/heart.csv. If missing, check DATASET_PATH env var. If both missing, raise clear error.
Feature Specification (src/features/feature_spec.py):

FeatureSpec: dataclass with numeric, binary, categorical lists.
FactoryConfig: dataclass controlling pipeline behavior (interactions, smoothing params).
Transformers (src/pipeline/preprocessor.py):

BloodPressureSplitter (Safe/Deterministic):
Input: blood_pressure ("120/80").
Output: systolic_bp, diastolic_bp (floats).
Logic: Split on /. If invalid/missing, return np.nan (handled by imputer).
CountryRiskTransformer (Learned):
Logic: Target encoding with Bayesian smoothing.
Formula: $w = \frac{n}{n + m}$ -> $value = w \times \mu_{country} + (1-w) \times \mu_{global}$.
$m$ (cfg.country_smoothing) defaults to 20.0.
Fit: Compute means on fit data. Transform: Map countries to values. Handle unknown countries by returning $\mu_{global}$.
Pipeline Factory (src/pipeline/pipeline_factory.py):

Returns sklearn.Pipeline:
BloodPressureSplitter
ColumnTransformer (Imputers + Scalers + Encoders + CountryRisk)
Numeric: SimpleImputer(median) + StandardScaler
Binary: SimpleImputer(most_frequent)
Categorical: OneHotEncoder(handle_unknown='ignore')
Estimator (LogReg or Tree)
3. Evaluation & Metrics (src/eval/evaluation.py)
Core Metrics:

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
Custom Metric: recall_at_precision_30

Logic:
Get precisions, recalls, thresholds from precision_recall_curve.
Filter indices where precision >= 0.30.
If no such index: Log warning, return threshold = 1.0 (Recall approx 0).
Else: Find index minimizing threshold (maximizing recall) among those meeting precision criteria.
Return recall at that index and the threshold value.
Plots:

Save strictly as png to artifacts/: roc_curve.png, pr_curve.png, confusion_matrix.png (at selected threshold).
4. Experiments (src/train/)
Stage 1 & 2 (run_experiments.py):

Loop through configs (Baseline LogReg, Tuned Tree).
Log to MLflow experiment "CardioSentinel_Experiments".
Tag: stage, model_type.
Save summary CSV to outputs/runs_summary.csv sorted by recall_at_precision_30.
Stage 3 (finalize_model.py):

Select best run from summary.
Retrain on FULL train+val set using SEED.
Evaluate on holdout Test set.
Save artifacts/model.joblib and artifacts/model_card.json.
model_card.json must include: training timestamp, metrics, threshold used for inference.
5. Inference & App
Inference (src/infer/predict.py):

load_model(): Loads joblib + JSON card.
predict_one(data_dict):
Validates input keys.
Runs pipeline predict_proba.
Applies stored threshold_at_precision_30.
Returns: { "risk_score": float, "is_high_risk": bool, "threshold_used": float }.
Streamlit App (app/streamlit_app.py):

Single page.
Load model on startup.
Form inputs for all features. blood_pressure as text input.
Show "High Risk" warning if risk_score > threshold.
Do not log user input.
6. Deployment (Dockerfile)
Base: python:3.10-slim.
Copy requirements.txt -> install.
Copy src/, app/, artifacts/ (if present) OR add build step to generate them.
Entrypoint:
CMD sh -c 'streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0'
This ensures compatibility with Render's dynamic $PORT.