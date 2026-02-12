"""CardioSentinel — Streamlit inference UI.

Loads the Phase 2 model artifacts and lets a user enter medical stats to
receive a risk score and high-risk flag.  No user data is logged or stored.
"""
from __future__ import annotations

import sys
import base64
import streamlit as st
from pathlib import Path


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

# Mapping from country to continent
_COUNTRY_CONTINENT = {
    "Argentina": "South America",
    "Australia": "Australia",
    "Brazil": "South America",
    "Canada": "North America",
    "China": "Asia",
    "Colombia": "South America",
    "France": "Europe",
    "Germany": "Europe",
    "India": "Asia",
    "Italy": "Europe",
    "Japan": "Asia",
    "New Zealand": "Australia",
    "Nigeria": "Africa",
    "South Africa": "Africa",
    "South Korea": "Asia",
    "Spain": "Europe",
    "Thailand": "Asia",
    "United Kingdom": "Europe",
    "United States": "North America",
    "Vietnam": "Asia",
}

# Mapping from country to hemisphere
_COUNTRY_HEMISPHERE = {
    "Argentina": "Southern Hemisphere",
    "Australia": "Southern Hemisphere",
    "Brazil": "Southern Hemisphere",
    "Canada": "Northern Hemisphere",
    "China": "Northern Hemisphere",
    "Colombia": "Northern Hemisphere",
    "France": "Northern Hemisphere",
    "Germany": "Northern Hemisphere",
    "India": "Northern Hemisphere",
    "Italy": "Northern Hemisphere",
    "Japan": "Northern Hemisphere",
    "New Zealand": "Southern Hemisphere",
    "Nigeria": "Northern Hemisphere",
    "South Africa": "Southern Hemisphere",
    "South Korea": "Northern Hemisphere",
    "Spain": "Northern Hemisphere",
    "Thailand": "Northern Hemisphere",
    "United Kingdom": "Northern Hemisphere",
    "United States": "Northern Hemisphere",
    "Vietnam": "Northern Hemisphere",
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

st.set_page_config(
    page_title="CardioSentinel",
    page_icon="app/assets/heart-wing.png",
)

# Header with icon
def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

icon_path = Path(__file__).parent / "assets" / "heart-wing.png"
icon_b64 = _img_to_base64(icon_path)

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom: 0.25rem;">
        <img src="data:image/png;base64,{icon_b64}" style="width:56px; height:56px;" />
        <div>
            <div style="font-size:3rem; font-weight:700; line-height:1.1;">CardioSentinel</div>
            <div style="color: #6b7280; margin-top: 0.25rem;">Heart attack risk screening demo</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")  # Blank line between title and disclaimer

st.markdown(
    """
    <div style="font-size: 0.85rem; color: #4b5563; line-height: 1.4;">
        <strong>Educational demo — not medical advice</strong><br/>
        This demonstrates a <em>screening</em> model optimized to maximize recall
        while maintaining a minimum precision.<br/>
        Risk labels reflect a fixed threshold selected during evaluation,
        not a clinical diagnosis or medical recommendation.<br/>        
        This demo uses a synthetic dataset with an elevated risk prevalence to illustrate screening trade-offs.
        <hr style="margin: 0.5rem 0;" />
        <em>Icon credit:</em> Heart-wing by Luch Phou — Flaticon
    </div>
    """,
    unsafe_allow_html=True,
)


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

# Country selection outside form to enable reactive updates
if "country" not in st.session_state:
    st.session_state["country"] = "Japan"  # A country where default stats were under threshold

def _on_country_change():
    """Update continent and hemisphere when country changes."""
    country = st.session_state["country"]
    st.session_state["continent"] = _COUNTRY_CONTINENT.get(country, "Asia")
    st.session_state["hemisphere"] = _COUNTRY_HEMISPHERE.get(country, "Northern Hemisphere")

# Initialize continent/hemisphere based on default country
if "continent" not in st.session_state:
    st.session_state["continent"] = _COUNTRY_CONTINENT["Japan"]
if "hemisphere" not in st.session_state:
    st.session_state["hemisphere"] = _COUNTRY_HEMISPHERE["Japan"]

# Country, Continent, Hemisphere on the same row
c_geo1, c_geo2, c_geo3 = st.columns(3)
country = c_geo1.selectbox(
    "Country",
    COUNTRIES,
    key="country",
    on_change=_on_country_change,
)
c_geo2.text_input("Continent", value=st.session_state["continent"], disabled=True)
c_geo3.text_input("Hemisphere", value=st.session_state["hemisphere"], disabled=True)

continent = st.session_state["continent"]
hemisphere = st.session_state["hemisphere"]

with st.expander("Why does country matter?"):
    st.write(
        "This model includes a country-level risk feature learned from historical data. "
        "It shifts the baseline risk upward or downward before individual factors are applied. "
        "Defaults with Japan show below threshold, but increase with the United States."
    )


with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    age = c1.number_input("Age", min_value=1, max_value=120, value=30)
    sex = c2.selectbox("Sex", SEXES, index=SEXES.index("Female"))
    diet = c3.selectbox("Diet", DIETS, index=DIETS.index("Healthy"))

    c4, c5, c6 = st.columns(3)
    cholesterol = c4.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=155)
    blood_pressure = c5.text_input("Blood Pressure (sys/dia)", value="108/68")
    blood_pressure = blood_pressure.replace(" ", "")
    heart_rate = c6.number_input("Heart Rate (bpm)", min_value=20, max_value=220, value=64)

    c7, c8, c9 = st.columns(3)
    bmi = c7.number_input("BMI", min_value=10.0, max_value=60.0, value=21.5, step=0.1)
    triglycerides = c8.number_input("Triglycerides (mg/dL)", min_value=10, max_value=1500, value=85)
    income = c9.number_input("Income (annual)", min_value=0, max_value=1_000_000, value=70_000, step=10_000)

    c10, c11, c12 = st.columns(3)
    exercise_hours = c10.number_input("Exercise hrs/week", min_value=0.0, max_value=40.0, value=8.0, step=0.5)
    physical_activity_days = c11.number_input("Physical activity days/week", min_value=0, max_value=7, value=5)
    sleep_hours = c12.number_input("Sleep hrs/day", min_value=0.0, max_value=24.0, value=8.0, step=0.5)

    c13, c14 = st.columns(2)
    stress_level = c13.slider("Stress level (1-10)", min_value=1, max_value=10, value=2)
    sedentary_hours = c14.number_input("Sedentary hrs/day", min_value=0.0, max_value=24.0, value=3.5, step=0.5)


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

    st.caption("🧭 Defaults reflect a generally healthy adult profile. Adjust values to explore how risk changes.")

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

    # Build input_dict FIRST
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

    # Then predict
    st.session_state["result"] = predict_one(input_dict, pipeline, model_card)

# ---- Results ---------------------------------------------------------
result = st.session_state.get("result")

if result is None:
    st.caption("Enter values and click Predict to see results.")
else:
    threshold = float(result["threshold_used"])
    score = float(result["risk_score"])
    margin = 0.05

    # Set risk levels
    if score < threshold:
        risk_label = "BELOW THRESHOLD"
        risk_state = "success"
        explanation = "Risk score is below the screening threshold."
    elif score < threshold + margin:
        risk_label = "SCREENING RANGE"
        risk_state = "warning"
        explanation = (
            "Risk score is just above the screening threshold. "
            "This model is optimized for sensitivity, so moderate profiles "
            "may fall in this range."
        )
    else:
        risk_label = "ABOVE THRESHOLD"
        risk_state = "error"
        explanation = (
            "Risk score exceeds the screening threshold used to prioritize recall."
        )

    with st.expander("Why might a profile fall near the threshold?"):
        st.markdown(
            """
            This model is designed for **screening**, not diagnosis.

            - It prioritizes identifying as many at-risk cases as possible (high recall).
            - The training dataset contains a relatively high baseline positive rate.
            - As a result, moderate or average profiles may score near the screening boundary.

            The model also includes **interaction terms**, meaning features influence one 
            another rather than acting independently. For example, stress level may interact 
            with exercise, BMI, or country-level baseline risk. The final score reflects the 
            **combined effect of all inputs**, not the impact of any single factor in isolation.

            The decision threshold is fixed from model evaluation and remains constant 
            during inference.
            """
        )


    # Display results
    st.divider()
    r1, r2 = st.columns(2)

    r1.metric("Risk score", f"{score * 100:.1f}%")

    if risk_state == "error":
        r2.error(f"**{risk_label}**")
    elif risk_state == "warning":
        r2.warning(f"**{risk_label}**")
    else:
        r2.success(f"**{risk_label}**")

    st.caption(f"Threshold used: {threshold:.4f}")
    st.caption(explanation)

