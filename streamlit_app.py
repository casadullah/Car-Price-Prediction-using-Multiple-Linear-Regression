import logging

import pandas as pd
import streamlit as st
from pathlib import Path

from src.predict import ModelLoadError, load_model_bundle, predict_price, validate_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "car_price_model.pkl"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

CUSTOM_CSS = """
<style>
    :root {
        --accent: #FF1E27;
    }
    .stApp {
        background-color: #0E0E10;
        color: #F2F2F2;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    .app-title {
        color: var(--accent) !important;
    }
    .stButton > button {
        background-color: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: background-color 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #cc1820;
        color: #FFFFFF;
    }
    .prediction-box {
        background-color: #1A1A1D;
        border: 1px solid var(--accent);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .prediction-box .price {
        color: var(--accent);
        font-size: 2.4rem;
        font-weight: 700;
    }
    .prediction-box .range {
        color: #AAAAAA;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    .footer {
        text-align: center;
        color: #888888;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2A2A2E;
        font-size: 0.9rem;
    }
    div[data-baseweb="select"] > div {
        background-color: #1A1A1D;
        border-color: #2A2A2E;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SAMPLE_INPUTS = {
    "Budget hatchback": {
        "brand": "Volkswagen", "body": "hatch", "engine_type": "Petrol",
        "registration": "yes", "mileage": 220, "engine_v": 1.6,
    },
    "Luxury SUV": {
        "brand": "Mercedes-Benz", "body": "crossover", "engine_type": "Diesel",
        "registration": "yes", "mileage": 60, "engine_v": 3.5,
    },
    "High-mileage van": {
        "brand": "Renault", "body": "van", "engine_type": "Diesel",
        "registration": "no", "mileage": 280, "engine_v": 2.0,
    },
}

DEFAULTS = SAMPLE_INPUTS["Budget hatchback"]


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(MODEL_PATH)


try:
    bundle = get_model_bundle()
    model_load_error = None
except ModelLoadError as exc:
    bundle = None
    model_load_error = str(exc)

st.markdown('<h1 class="app-title">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)

if model_load_error:
    st.error(
        f"The prediction model could not be loaded: {model_load_error}\n\n"
        "The app cannot make predictions until a trained model is available."
    )
    st.stop()

valid_ranges = bundle.get("valid_ranges", {})
mileage_range = valid_ranges.get("Mileage", (0, 300))
enginev_range = valid_ranges.get("EngineV", (0.6, 6.5))

st.markdown(
    f"""
This app estimates a used car's market price using a **{bundle.get('model_name', 'machine learning')}**
model, selected after comparing Linear Regression, Random Forest, and Gradient
Boosting on cross-validated performance. Enter the car's details below and
click **Predict Price** to get an instant estimate with an approximate range.

*Valid operating range: mileage {mileage_range[0]:,.0f}-{mileage_range[1]:,.0f} thousand km,
engine volume {enginev_range[0]:.1f}L-{enginev_range[1]:.1f}L. Predictions outside this range
are extrapolations and less reliable.*
"""
)

st.divider()

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.write("**Try a sample car:**")
sample_cols = st.columns(len(SAMPLE_INPUTS))
for col, (label, values) in zip(sample_cols, SAMPLE_INPUTS.items()):
    if col.button(label, width="stretch"):
        for key, value in values.items():
            st.session_state[key] = value
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", bundle["brands"], key="brand")
    body = st.selectbox("Body Type", bundle["bodies"], key="body")
    engine_type = st.selectbox("Engine Type", bundle["engine_types"], key="engine_type")

with col2:
    mileage = st.number_input(
        "Mileage (thousand km)", min_value=0, max_value=2000, step=1, key="mileage"
    )
    engine_v = st.slider(
        "Engine Volume (L)", min_value=0.5, max_value=10.0, step=0.1, key="engine_v"
    )
    registration = st.selectbox("Registered", bundle["registrations"], key="registration")

st.write("")
predict_clicked = st.button("Predict Price")

if predict_clicked:
    try:
        validation = validate_input(
            brand, body, engine_type, registration, mileage, engine_v,
            valid_ranges=valid_ranges,
        )
        if not validation.is_valid:
            for error in validation.errors:
                st.error(error)
        else:
            result = predict_price(
                bundle, brand, body, engine_type, registration, mileage, engine_v
            )
            for warning in result.warnings:
                st.warning(warning)

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div>Estimated Price</div>
                    <div class="price">${result.price:,.0f}</div>
                    <div class="range">Likely range (~95%): ${result.price_low:,.0f} - ${result.price_high:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except ValueError as exc:
        st.error(f"Could not generate a prediction: {exc}")
    except Exception:
        logger.exception("Unexpected error while predicting")
        st.error("Something went wrong while generating the prediction. Please check your inputs and try again.")

with st.expander("📊 Model Performance"):
    st.write(f"**Selected model:** {bundle.get('model_name', 'unknown')}")

    cv_results = bundle.get("cv_results", {})
    if cv_results:
        st.write("**5-fold cross-validation (training set, R² on log-price):**")
        cv_df = pd.DataFrame(
            [
                {"Model": name, "R² mean": res["r2_mean"], "R² std": res["r2_std"]}
                for name, res in cv_results.items()
            ]
        )
        st.dataframe(cv_df, hide_index=True, width="stretch")

    test_metrics = bundle.get("test_metrics")
    if test_metrics:
        st.write("**Held-out test set (price scale):**")
        metrics_df = pd.DataFrame(
            [
                {"Metric": "R²", "Value": f"{test_metrics['r2']:.3f}"},
                {"Metric": "RMSE", "Value": f"${test_metrics['rmse']:,.0f}"},
                {"Metric": "MAE", "Value": f"${test_metrics['mae']:,.0f}"},
                {"Metric": "MAPE", "Value": f"{test_metrics['mape']:.1f}%"},
            ]
        )
        st.dataframe(metrics_df, hide_index=True, width="stretch")

    feature_importance = bundle.get("feature_importance")
    if feature_importance:
        st.write("**What matters most for price:**")
        fi_df = pd.DataFrame(feature_importance).head(10).set_index("Feature")
        st.bar_chart(fi_df["Importance"])

st.markdown('<div class="footer">Built by Vexanex Digital Solutions</div>', unsafe_allow_html=True)
