import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
brands = bundle["brands"]
bodies = bundle["bodies"]
engine_types = bundle["engine_types"]

st.markdown('<h1 class="app-title">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    """
This app estimates a used car's market price using a **Multiple Linear
Regression** model trained on ~4,000 real car sale listings. Enter the
car's details below and click **Predict Price** to get an instant estimate.
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands)
    body = st.selectbox("Body Type", bodies)
    engine_type = st.selectbox("Engine Type", engine_types)

with col2:
    mileage = st.number_input("Mileage (thousand km)", min_value=0, max_value=1000, value=150, step=1)
    engine_v = st.slider("Engine Volume (L)", min_value=0.6, max_value=6.5, value=2.0, step=0.1)
    registration = st.selectbox("Registered", ["yes", "no"])

st.write("")
predict_clicked = st.button("Predict Price")

if predict_clicked:
    row = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
    row["Mileage"] = mileage
    row["EngineV"] = engine_v

    brand_col = f"Brand_{brand}"
    if brand_col in row.columns:
        row[brand_col] = 1

    body_col = f"Body_{body}"
    if body_col in row.columns:
        row[body_col] = 1

    engine_col = f"Engine Type_{engine_type}"
    if engine_col in row.columns:
        row[engine_col] = 1

    if registration == "yes":
        row["Registration_yes"] = 1

    scaled_row = scaler.transform(row)
    log_price_pred = model.predict(scaled_row)[0]
    price_pred = np.exp(log_price_pred)

    st.markdown(
        f"""
        <div class="prediction-box">
            <div>Estimated Price</div>
            <div class="price">${price_pred:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">Built by Vexanex Digital Solutions</div>', unsafe_allow_html=True)
