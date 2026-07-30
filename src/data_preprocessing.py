"""Data cleaning, encoding, and feature engineering for the car price model.

Reproduces the cleaning pipeline from notebook/Practical_Example.ipynb:
drop rows with missing values, cap outliers on Mileage/EngineV/Year, log-transform
the target, one-hot encode categoricals, and drop `Year` (removed for
multicollinearity per the notebook's VIF analysis).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_COLUMNS_TO_DROP = ["Model"]

# One-hot dummy columns produced by pd.get_dummies(..., drop_first=True).
# Baseline (dropped) categories: Brand=Audi, Body=crossover, Engine Type=Diesel,
# Registration=no.
FEATURE_COLUMNS = [
    "Mileage", "EngineV",
    "Brand_BMW", "Brand_Mercedes-Benz", "Brand_Mitsubishi", "Brand_Renault",
    "Brand_Toyota", "Brand_Volkswagen",
    "Body_hatch", "Body_other", "Body_sedan", "Body_vagon", "Body_van",
    "Engine Type_Gas", "Engine Type_Other", "Engine Type_Petrol",
    "Registration_yes",
]

BRAND_OPTIONS = ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"]
BODY_OPTIONS = ["crossover", "hatch", "other", "sedan", "vagon", "van"]
ENGINE_TYPE_OPTIONS = ["Diesel", "Gas", "Other", "Petrol"]
REGISTRATION_OPTIONS = ["yes", "no"]

REQUIRED_RAW_COLUMNS = [
    "Brand", "Price", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw car sales CSV.

    Args:
        path: Path to the raw CSV file.

    Returns:
        The raw dataframe, unmodified.

    Raises:
        FileNotFoundError: If the CSV does not exist at `path`.
        ValueError: If the CSV is missing expected columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    try:
        raw_data = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV at {path}: {exc}") from exc

    missing = set(REQUIRED_RAW_COLUMNS) - set(raw_data.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")

    logger.info("Loaded raw data: %d rows from %s", len(raw_data), path)
    return raw_data


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: drop missing values and cap outliers.

    Applies the same quantile-based outlier removal as the original notebook:
    Mileage capped at its 99th percentile, EngineV capped at 6.5, Year floored
    at its 1st percentile (computed on the mileage/EngineV-filtered subset).

    Args:
        raw_data: Output of `load_raw_data`.

    Returns:
        A cleaned dataframe with `Model` dropped and outlier rows removed,
        index reset.
    """
    data = raw_data.drop(columns=RAW_COLUMNS_TO_DROP, errors="ignore")
    data_no_mv = data.dropna(axis=0)
    logger.info("Dropped %d rows with missing values", len(data) - len(data_no_mv))

    mileage_q99 = data_no_mv["Mileage"].quantile(0.99)
    data_2 = data_no_mv[data_no_mv["Mileage"] < mileage_q99]

    data_3 = data_2[data_2["EngineV"] < 6.5]

    year_q01 = data_3["Year"].quantile(0.01)
    data_4 = data_3[data_3["Year"] > year_q01]

    data_cleaned = data_4.reset_index(drop=True)
    logger.info("Cleaned data: %d rows remaining after outlier removal", len(data_cleaned))
    return data_cleaned


def engineer_features(data_cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the model-ready feature matrix and log-price target.

    Log-transforms Price, drops Year (multicollinear with Mileage/EngineV per
    VIF analysis), one-hot encodes categoricals with `drop_first=True`, and
    aligns the result to `FEATURE_COLUMNS`.

    Args:
        data_cleaned: Output of `clean_data`.

    Returns:
        Tuple of (features, targets) where targets is log(Price).
    """
    data = data_cleaned.copy()
    data["log_price"] = np.log(data["Price"])
    data = data.drop(columns=["Price", "Year"])

    data_with_dummies = pd.get_dummies(data, drop_first=True)

    missing_cols = set(FEATURE_COLUMNS) - set(data_with_dummies.columns)
    for col in missing_cols:
        data_with_dummies[col] = False

    features = data_with_dummies[FEATURE_COLUMNS]
    targets = data_with_dummies["log_price"]
    return features, targets


def compute_valid_ranges(data_cleaned: pd.DataFrame) -> dict:
    """Compute the training data's valid operating ranges for numeric inputs.

    Used both for documentation ("trained on cars 0-X mileage...") and for
    flagging out-of-distribution prediction requests at inference time.

    Args:
        data_cleaned: Output of `clean_data`.

    Returns:
        Dict mapping column name to (min, max) tuples, plus the price range.
    """
    return {
        "Mileage": (float(data_cleaned["Mileage"].min()), float(data_cleaned["Mileage"].max())),
        "EngineV": (float(data_cleaned["EngineV"].min()), float(data_cleaned["EngineV"].max())),
        "Year": (float(data_cleaned["Year"].min()), float(data_cleaned["Year"].max())),
        "Price": (float(data_cleaned["Price"].min()), float(data_cleaned["Price"].max())),
    }


def load_and_prepare(path: Path) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Convenience wrapper: load, clean, and featurize the raw dataset in one call.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Tuple of (features, targets, valid_ranges).
    """
    raw_data = load_raw_data(path)
    data_cleaned = clean_data(raw_data)
    features, targets = engineer_features(data_cleaned)
    valid_ranges = compute_valid_ranges(data_cleaned)
    return features, targets, valid_ranges
