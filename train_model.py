"""
Trains the Multiple Linear Regression car price model following the exact
cleaning / feature-engineering steps from notebook/Practical_Example.ipynb,
then saves the fitted model + scaler + metadata to model/car_price_model.pkl
so streamlit_app.py can load it instantly instead of retraining every run.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Car_Sales_Raw_Data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "car_price_model.pkl"

FEATURE_COLUMNS = [
    "Mileage", "EngineV",
    "Brand_BMW", "Brand_Mercedes-Benz", "Brand_Mitsubishi", "Brand_Renault",
    "Brand_Toyota", "Brand_Volkswagen",
    "Body_hatch", "Body_other", "Body_sedan", "Body_vagon", "Body_van",
    "Engine Type_Gas", "Engine Type_Other", "Engine Type_Petrol",
    "Registration_yes",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(path)
    data = raw_data.drop(["Model"], axis=1)
    data_no_mv = data.dropna(axis=0)

    q = data_no_mv["Price"].quantile(0.99)
    data_1 = data_no_mv[data_no_mv["Price"] < q]  # noqa: F841 (kept for parity with notebook)

    q = data_no_mv["Mileage"].quantile(0.99)
    data_2 = data_no_mv[data_no_mv["Mileage"] < q]

    data_3 = data_2[data_2["EngineV"] < 6.5]

    q = data_3["Year"].quantile(0.01)
    data_4 = data_3[data_3["Year"] > q]

    return data_4.reset_index(drop=True)


def build_features(data_cleaned: pd.DataFrame):
    data_cleaned = data_cleaned.copy()
    data_cleaned["log_price"] = np.log(data_cleaned["Price"])
    data_cleaned = data_cleaned.drop(["Price"], axis=1)
    data_no_multicollinearity = data_cleaned.drop(["Year"], axis=1)
    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

    cols = ["log_price"] + FEATURE_COLUMNS
    data_preprocessed = data_with_dummies[cols]

    targets = data_preprocessed["log_price"]
    features = data_preprocessed.drop(["log_price"], axis=1)
    return features, targets


def main():
    data_cleaned = load_and_clean(DATA_PATH)
    features, targets = build_features(data_cleaned)

    scaler = StandardScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=365
    )

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    train_r2 = reg.score(x_train, y_train)
    test_r2 = reg.score(x_test, y_test)
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test R^2:  {test_r2:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": reg,
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "brands": ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"],
            "bodies": ["crossover", "hatch", "other", "sedan", "vagon", "van"],
            "engine_types": ["Diesel", "Gas", "Other", "Petrol"],
            "registrations": ["yes", "no"],
            "mileage_range": (0, 980),
            "enginev_range": (0.6, 6.5),
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
