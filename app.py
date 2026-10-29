import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from html import escape

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "best_tuned_model.pkl"
ILLUSTRATION_PATH = ROOT / "assets" / "doctor_illustration.png"

# ------------------------------------------------------------
# Styling: pale blue healthcare dashboard
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #071a42;
        --muted: #385681;
        --blue: #0862f5;
        --blue-dark: #0754df;
        --line: #d6e7ff;
        --panel: rgba(255,255,255,.83);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 82% 14%, rgba(211,232,255,.62), transparent 34%),
            linear-gradient(135deg, #f2f8ff 0%, #eaf4ff 52%, #f5faff 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        max-width: 1580px;
        padding: 1.05rem 2.1rem 1.7rem;
    }
    h1, h2, h3, p, label, span, div { letter-spacing: -0.015em; }
    .brand-row {
        display: flex; align-items: center; gap: 18px;
        margin: 0 0 18px; min-height: 76px;
    }
    .brand-shield {
        width: 72px; height: 72px; flex: 0 0 72px;
        display: grid; place-items: center;
        font-size: 42px;
        filter: drop-shadow(0 5px 8px rgba(12,92,210,.14));
    }
    .brand-title {
        color: var(--ink); font-size: clamp(27px, 2.45vw, 43px);
        line-height: 1.12; font-weight: 800; margin: 0;
    }
    .brand-title .accent { color: var(--blue); }
    .brand-subtitle { color: var(--muted); font-size: 16px; margin-top: 8px; }
    .brand-tagline {
        margin-left: auto; text-align: center; color: #173e7d;
        font-size: 14px; line-height: 1.35; padding: 0 10px 9px;
        border-bottom: 3px solid var(--blue); align-self: center;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border: 1px solid rgba(205,226,255,.95) !important;
        border-radius: 20px !important;
        box-shadow: 0 9px 28px rgba(38,91,157,.055);
    }
    .section-heading {
        display:flex; align-items:center; gap:15px; margin: 0 0 5px;
        color:var(--ink); font-size:24px; font-weight:800;
    }
    .section-icon {
        width:54px;height:54px; flex:0 0 54px;border-radius:50%;
        background:linear-gradient(145deg,#e0efff,#cce2ff);
        display:grid;place-items:center;color:var(--blue);font-size:26px;
    }
    .section-description { color:var(--muted); font-size:15px; margin:0 0 16px 69px; }
    [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important; font-weight: 650 !important;
        font-size: 14px !important; margin-bottom: 6px !important;
    }
    [data-testid="stSlider"] { padding-top: 0.2rem; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        border-color: var(--blue) !important;
        background: var(--blue) !important;
        box-shadow: 0 0 0 4px rgba(8,98,245,.12);
    }
    [data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--muted); }
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        border: 1px solid #c5dcff !important;
        background: linear-gradient(110deg,#f5f9ff,#edf5ff) !important;
        border-radius: 10px !important;
        min-height: 43px;
    }
    [data-baseweb="select"] * { color: var(--ink); }
    [data-testid="stNumberInput"] input { color: var(--ink) !important; }
    [data-testid="stRadio"] label { color: var(--ink) !important; }
    [data-testid="stRadio"] [data-checked="true"] div { border-color: var(--blue) !important; }
    [data-testid="stRadio"] [data-checked="true"] div div { background: var(--blue) !important; }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg,#0862f5,#0758ee) !important;
        color: white !important; border: 0 !important; border-radius: 11px !important;
        padding: 0.72rem 1.25rem !important; min-height: 48px;
        font-size: 15px !important; font-weight: 750 !important;
        box-shadow: 0 6px 14px rgba(8,98,245,.16);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 9px 18px rgba(8,98,245,.23);
        transform: translateY(-1px);
    }
    .result-card {
        display:flex; align-items:center; justify-content:space-between; gap:22px;
        padding: 14px 18px; border:1px solid #d7e8ff; border-radius:17px;
        background:linear-gradient(110deg,rgba(255,255,255,.93),rgba(246,251,255,.95));
        margin: 8px 0 10px;
    }
    .result-left { display:flex; align-items:center; gap:22px; min-width:0; }
    .money-icon {
        width:92px;height:88px;border-radius:16px;display:grid;place-items:center;
        color:var(--blue);font-size:40px;background:linear-gradient(145deg,#e4f0ff,#d2e6ff);
        flex:0 0 auto;
    }
    .result-label { color:var(--blue);font-size:17px;font-weight:700;margin-bottom:1px; }
    .result-value { color:var(--blue);font-size:clamp(31px,3vw,47px);line-height:1.15;font-weight:800; }
    .success-box {
        min-width: 34%; padding: 17px 22px; border-radius:16px;
        background:linear-gradient(100deg,#effaf4,#e7f7ee);
        display:flex;align-items:center;gap:15px;
    }
    .success-check {
        display:grid;place-items:center;width:42px;height:42px;border-radius:50%;
        background:#13863d;color:#fff;font-size:27px;font-weight:800;flex:0 0 auto;
    }
    .success-title { color:#087c32;font-size:16px;font-weight:800;margin-bottom:4px; }
    .success-copy { color:var(--muted);font-size:14px; }
    .details-title {
        display:flex;align-items:center;gap:12px;font-size:19px;font-weight:800;
        color:var(--ink);padding:2px 0 10px;
    }
    [data-testid="stDataFrame"] {
        border:1px solid #d3e5ff; border-radius:10px; overflow:hidden;
    }
    [data-testid="stDataFrame"] * { font-size:13px; }
    .fine-print { color:#5c7397;font-size:11px;margin-top:9px; }
    @media (max-width: 900px) {
        .block-container { padding: .8rem .8rem 1.2rem; }
        .brand-row { gap:10px; }
        .brand-shield { width:48px;height:48px;flex-basis:48px;font-size:31px; }
        .brand-tagline { display:none; }
        .brand-subtitle { font-size:13px; }
        .section-heading { font-size:20px; }
        .result-card { align-items:flex-start; flex-direction:column; }
        .success-box { width:100%; box-sizing:border-box; }
        .money-icon { width:70px;height:70px;font-size:32px; }
    }

    /* Subtle outlines around each customer input so controls are distinct
       from the pale page background. */
    [data-testid="stSlider"],
    [data-testid="stSelectbox"],
    [data-testid="stNumberInput"],
    [data-testid="stRadio"] {
        box-sizing: border-box;
        border: 1px solid #d4e2f3 !important;
        border-radius: 11px !important;
        background: rgba(255, 255, 255, 0.58) !important;
        padding: 10px 12px 8px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 2px rgba(31, 76, 135, 0.025);
    }

    /* Keep the actual select and number fields crisp inside their outlined
       field containers without creating a heavy double border. */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        border: 1px solid #c8d9ee !important;
        border-radius: 8px !important;
        background: #f8fbff !important;
    }

    /* Ensure the labels remain readable against the light field background. */
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
    [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
        color: #071a42 !important;
    }

    /* Give the slider track a defined, subtle boundary. */
    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 3px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="brand-row">
      <div class="brand-shield">🛡️</div>
      <div>
        <div class="brand-title">Medical Insurance <span class="accent">Charges Predictor</span></div>
        <div class="brand-subtitle">Estimate your medical insurance charges based on key health and lifestyle factors.</div>
      </div>
      <div class="brand-tagline">Better Insights<br>for a Healthier Tomorrow</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Input form + local illustration
# ------------------------------------------------------------
input_col, art_col = st.columns([2.25, 0.95], gap="medium")

with input_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-heading"><span class="section-icon">♙</span><span>Enter Customer Details</span></div>
            <p class="section-description">Provide the customer information below to estimate medical insurance charges.</p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("insurance_form", clear_on_submit=False):
            left, right = st.columns(2, gap="large")
            with left:
                age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
                sex = st.selectbox("Sex", ["male", "female"])
                bmi = st.number_input(
                    "BMI (Body Mass Index)", min_value=10.0, max_value=60.0,
                    value=25.5, step=0.1, format="%.2f"
                )
            with right:
                children = st.selectbox("Number of Children", list(range(11)), index=1)
                smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
                region = st.selectbox(
                    "Region", ["northeast", "northwest", "southeast", "southwest"]
                )
            submitted = st.form_submit_button("▦   Predict Insurance Charges   →", use_container_width=False)

with art_col:
    if ILLUSTRATION_PATH.exists():
        st.image(str(ILLUSTRATION_PATH), use_container_width=True)
    else:
        st.markdown(
            '<div style="height:440px;display:grid;place-items:center;font-size:100px;">🩺</div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# Prediction result
# ------------------------------------------------------------
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
}])

if "prediction" not in st.session_state:
    if model is not None:
        try:
            st.session_state.prediction = float(model.predict(input_data)[0])
        except Exception:
            st.session_state.prediction = None
    else:
        st.session_state.prediction = None

if submitted:
    if model is None:
        st.error("Model file not found. Place best_tuned_model.pkl next to app.py.")
    else:
        try:
            st.session_state.prediction = float(model.predict(input_data)[0])
        except Exception as exc:
            st.session_state.prediction = None
            st.error(f"Prediction failed: {exc}")

st.markdown(
    """
    <div class="section-heading" style="margin-top:8px;">
      <span class="section-icon">▤</span><span>Prediction Result</span>
    </div>
    <p class="section-description">Estimated medical insurance charges based on the provided details.</p>
    """,
    unsafe_allow_html=True,
)

prediction = st.session_state.get("prediction")
if prediction is not None:
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-left">
            <div class="money-icon">▤</div>
            <div>
              <div class="result-label">Predicted Insurance Charges</div>
              <div class="result-value">${prediction:,.2f}</div>
            </div>
          </div>
          <div class="success-box">
            <div class="success-check">✓</div>
            <div>
              <div class="success-title">Prediction generated</div>
              <div class="success-copy">Based on the input details provided.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Enter the customer details above and select Predict Insurance Charges to see the result.")

with st.container(border=True):
    st.markdown('<div class="details-title">▤ &nbsp; View Customer Details</div>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True, hide_index=False)

st.markdown(
    '<div class="fine-print">For educational demonstration only. Predictions depend on the model and its training data.</div>',
    unsafe_allow_html=True,
)
