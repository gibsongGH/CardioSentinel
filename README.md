# CardioSentinel

**End-to-end machine learning system for heart attack risk screening**, optimized for high recall under a fixed precision constraint and deployed as an interactive web application.

🔗 **Live demo:** *(add Render URL here)*  
💻 **Source code:** https://github.com/gibsongGH/CardioSentinel

> ⚠️ **Disclaimer:** This project is an educational demonstration and is **not** medical advice.

---

## Overview

CardioSentinel demonstrates how to design, evaluate, and deploy a risk-screening model where **operating-point decisions matter as much as model choice**.

In screening contexts, false negatives can be costly, but excessive false positives undermine trust. Rather than optimizing accuracy alone, this project selects a model based on **maximum recall subject to a minimum precision constraint**, then deploys that model with a **fixed inference threshold** for consistent real-world behavior.

---

## Modeling approach

Multiple model families were evaluated, including:

- Logistic Regression (with and without interaction features)
- Tree-based models (HistGradientBoosting / XGBoost)

Standard metrics (accuracy, ROC-AUC, PR-AUC) were logged, but **model selection was driven by a custom operating-point metric**.

### Key metric: `recall_at_precision_40`

> *“If we lower the decision threshold until at least 40% of flagged patients are truly at risk (precision ≥ 0.40), what fraction of all actual at-risk patients do we catch?”*

This metric directly encodes the screening trade-off.

---

## Final model (Phase 2)

- **Model:** Logistic Regression with interaction features  
- **Selection criterion:** Highest `recall_at_precision_40`  
- **Precision floor:** 0.40  
- **Fixed inference threshold:** `0.3807` (derived during model selection and frozen for inference) 
- **Inference behavior:** Threshold is **frozen** and not recomputed at prediction time  

Model artifacts, evaluation plots, and a machine-readable **model card** are packaged under `artifacts/`.

---

## System architecture

- Reproducible sklearn Pipelines for preprocessing and modeling
- Custom transformers (blood pressure parsing, country risk encoding)
- Deterministic train / validation / test split with fixed seed
- MLflow experiment tracking for Phase 1 comparisons
- Model card generation for transparency and safe use
- Streamlit-based inference UI (Phase 3)
- Dockerized for deployment
- Hosted publicly via Render

---

## Local execution

Choose one of the following options, then open http://localhost:8501

### Streamlit

'''bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py

### Docker

'''bash
docker build -t cardiosentinel .
docker run -p 8501:8501 cardiosentinel

---

## Project structure

artifacts/                  <- final model, model card, evaluation plots
app/streamlit_app.py        <- Streamlit inference UI
src/
  data/load_data.py         <- CSV loader with env-var fallback
  features/feature_spec.py  <- FeatureSpec + FactoryConfig
  pipeline/                 <- preprocessing + pipeline assembly
  eval/evaluation.py        <- metrics, threshold logic, plots
  train/run_experiments.py  <- Phase 1 experiment runner
  train/finalize_model.py   <- Phase 2 final model packaging
  infer/predict.py          <- Phase 3 inference logic
  utils/                    <- seeding and MLflow helpers

---

## Data

- Source dataset:
https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset
(Public, clean educational dataset used for demonstration purposes)

Place heart.csv in data/, or set:
export DATASET_PATH=/abs/path/to/heart.csv

- Data quality note:
Basic data validation checks (missing values, duplicates, outliers) were performed prior to modeling. The dataset was sufficiently clean to proceed without extensive exploratory analysis, allowing the project to focus on model selection, thresholding, and deployment.

---

## Reproducibility

- Fixed random seed: SEED = 42
- Stratified 70 / 15 / 15 train / validation / test split
- Split configuration and seed logged for all experiments

---

## Limitations & ethics

- Educational demonstration only; not medical advice
- Performance depends on threshold choice
- Risk scores are not calibrated probabilities
- Trained on a single dataset
- False positives may cause unnecessary concern
- False negatives may miss at-risk individuals

---

## Development note (AI agents)

Parts of this project were developed with the assistance of AI coding agents under tightly scoped prompts. Architectural decisions, metric definitions, model selection criteria, and deployment strategy were designed and validated manually. All generated code was reviewed, modified, and integrated by the author.


