"""CardioSentinel — Streamlit inference UI.

Loads the Phase 2 model artifacts and lets a user enter medical stats to
receive a risk score and high-risk flag.  No user data is logged or stored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when running via `streamlit run app/streamlit_app.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.infer.predict import RAW_INPUT_COLUMNS, load_model, predict_one

# ------------------------------------------------------------------
# Constants for the input form
# ------------------------------------------------------------------

COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Canada", "China", "Colombia",
    "France", "Germany", "India", "Italy", "Japan", "New Zealand",
    "Nigeria", "South Africa", "South Korea", "Spain", "Thailand",
    "United Kingdom", "United States", "Vietnam",
]

CONTINENTS = [
    "Africa", "Asia", "Australia", "Europe", "North America", "South America",
]

HEMISPHERES = ["Northern Hemisphere", "Southern Hemisphere"]

DIETS = ["Average", "Healthy", "Unhealthy"]

SEXES = ["Male", "Female"]

# Mapping from continent to reasonable default hemisphere for convenience
_CONTINENT_HEMISPHERE = {
    "Africa": "Northern Hemisphere",
    "Asia": "Northern Hemisphere",
    "Australia": "Southern Hemisphere",
    "Europe": "Northern Hemisphere",
    "North America": "Northern Hemisphere",
    "South America": "Southern Hemisphere",
}


# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------

@st.cache_resource
def _load_artifacts():
    """Load model + card once and cache across reruns."""
    artifacts_dir = _PROJECT_ROOT / "artifacts"
    return load_model(artifacts_dir)


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

st.set_page_config(page_title="CardioSentinel", page_icon=":anatomical_heart:")

st.title("CardioSentinel")
st.caption("Educational demo. Not medical advice.")

# Attempt to load model artifacts
try:
    pipeline, model_card = _load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run:\n\n"
        "```\npython -m src.train.finalize_model\n```"
    )
    st.stop()

# ---- Model info panel ------------------------------------------------
with st.expander("Model info", expanded=False):
    col1, col2 = st.columns(2)
    col1.metric("Model", model_card.get("model_name", "N/A"))
    col2.metric("Precision floor", f'{model_card.get("precision_floor", "N/A")}')
    col3, col4 = st.columns(2)
    col3.metric("Decision threshold", f'{model_card.get("threshold", "N/A"):.4f}')
    col4.metric("Created", model_card.get("created_at", "N/A")[:10])

# ---- Input form ------------------------------------------------------
st.subheader("Enter patient data")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    age = c1.number_input("Age", min_value=1, max_value=120, value=50)
    sex = c2.selectbox("Sex", SEXES)
    diet = c3.selectbox("Diet", DIETS)

    c4, c5, c6 = st.columns(3)
    cholesterol = c4.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=200)
    blood_pressure = c5.text_input("Blood Pressure (sys/dia)", value="120/80")
    heart_rate = c6.number_input("Heart Rate (bpm)", min_value=20, max_value=220, value=72)

    c7, c8, c9 = st.columns(3)
    bmi = c7.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.1)
    triglycerides = c8.number_input("Triglycerides (mg/dL)", min_value=10, max_value=1500, value=150)
    income = c9.number_input("Income (annual)", min_value=0, max_value=1_000_000, value=60000, step=10000)

    c10, c11, c12 = st.columns(3)
    exercise_hours = c10.number_input("Exercise hrs/week", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
    physical_activity_days = c11.number_input("Physical activity days/week", min_value=0, max_value=7, value=3)
    sleep_hours = c12.number_input("Sleep hrs/day", min_value=0, max_value=24, value=7)

    c13, c14, c15 = st.columns(3)
    stress_level = c13.slider("Stress level (1-10)", min_value=1, max_value=10, value=5)
    sedentary_hours = c14.number_input("Sedentary hrs/day", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
    country = c15.selectbox("Country", COUNTRIES, index=COUNTRIES.index("United States"))

    c16, c17 = st.columns(2)
    continent = c16.selectbox("Continent", CONTINENTS, index=CONTINENTS.index("North America"))
    hemisphere = c17.selectbox("Hemisphere", HEMISPHERES)

    st.markdown("**Medical history**")
    h1, h2, h3, h4 = st.columns(4)
    diabetes = int(h1.checkbox("Diabetes"))
    family_history = int(h2.checkbox("Family history"))
    smoking = int(h3.checkbox("Smoking"))
    obesity = int(h4.checkbox("Obesity"))

    h5, h6, h7 = st.columns(3)
    alcohol_consumption = int(h5.checkbox("Alcohol consumption"))
    previous_heart_problems = int(h6.checkbox("Previous heart problems"))
    medication_use = int(h7.checkbox("Medication use"))

    submitted = st.form_submit_button("Predict")

# ---- Prediction ------------------------------------------------------
if submitted:
    # Validate blood_pressure format
    bp_warning = False
    bp_value = blood_pressure.strip()
    if "/" not in bp_value:
        bp_warning = True
    else:
        parts = bp_value.split("/")
        if len(parts) != 2:
            bp_warning = True
        else:
            try:
                int(parts[0])
                int(parts[1])
            except ValueError:
                bp_warning = True

    if bp_warning:
        st.warning(
            "Blood pressure format looks invalid (expected e.g. '120/80'). "
            "The model will still run but may treat this as missing data."
        )

    input_dict = {
        "age": age,
        "sex": sex,
        "cholesterol": cholesterol,
        "blood_pressure": bp_value,
        "heart_rate": heart_rate,
        "diabetes": diabetes,
        "family_history": family_history,
        "smoking": smoking,
        "obesity": obesity,
        "alcohol_consumption": alcohol_consumption,
        "exercise_hours_per_week": exercise_hours,
        "diet": diet,
        "previous_heart_problems": previous_heart_problems,
        "medication_use": medication_use,
        "stress_level": stress_level,
        "sedentary_hours_per_day": sedentary_hours,
        "income": income,
        "bmi": bmi,
        "triglycerides": triglycerides,
        "physical_activity_days_per_week": physical_activity_days,
        "sleep_hours_per_day": sleep_hours,
        "country": country,
        "continent": continent,
        "hemisphere": hemisphere,
    }

    try:
        result = predict_one(input_dict, pipeline, model_card)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Display results
    st.divider()
    r1, r2 = st.columns(2)

    risk_pct = result["risk_score"] * 100
    r1.metric("Risk score", f"{risk_pct:.1f}%")

    if result["is_high_risk"]:
        r2.error("**HIGH RISK**")
    else:
        r2.success("**Lower Risk**")

    st.caption(f"Threshold used: {result['threshold_used']:.4f}")
