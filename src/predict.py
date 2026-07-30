"""Prediction logic: input validation, out-of-distribution flagging, and
point/interval price estimates for a loaded model bundle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    BODY_OPTIONS,
    BRAND_OPTIONS,
    ENGINE_TYPE_OPTIONS,
    FEATURE_COLUMNS,
    REGISTRATION_OPTIONS,
)

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model bundle cannot be loaded."""


@dataclass
class ValidationResult:
    """Outcome of validating a single prediction request.

    Attributes:
        errors: Hard failures (invalid category, wrong type) that block prediction.
        warnings: Soft flags (value outside the training distribution) that
            do not block prediction but should be surfaced to the user.
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass
class PredictionResult:
    """A completed prediction, including an approximate interval and any flags."""

    price: float
    price_low: float
    price_high: float
    warnings: List[str]


def load_model_bundle(path: Path) -> dict:
    """Load the trained model bundle produced by `src/train.py`.

    Args:
        path: Path to the joblib-serialized bundle.

    Returns:
        Dict containing at least "model", "scaler", "feature_columns",
        "valid_ranges", "metrics", and "feature_importance".

    Raises:
        ModelLoadError: If the file is missing or cannot be deserialized.
    """
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}. Run `python -m src.train` first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # joblib/pickle can raise many exception types
        raise ModelLoadError(f"Could not load model bundle from {path}: {exc}") from exc

    required_keys = {"model", "scaler", "feature_columns"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ModelLoadError(f"Model bundle at {path} is missing keys: {sorted(missing)}")

    return bundle


def validate_input(
    brand: str,
    body: str,
    engine_type: str,
    registration: str,
    mileage: float,
    engine_v: float,
    valid_ranges: Optional[dict] = None,
) -> ValidationResult:
    """Validate a raw prediction request before it is fed to the model.

    Checks categorical fields against the known training categories (hard
    errors) and flags numeric fields that fall outside the training data's
    observed range (soft warnings, since the model can still produce a
    number but it is extrapolating).

    Args:
        brand: One of BRAND_OPTIONS.
        body: One of BODY_OPTIONS.
        engine_type: One of ENGINE_TYPE_OPTIONS.
        registration: One of REGISTRATION_OPTIONS.
        mileage: Mileage in thousand km.
        engine_v: Engine volume in liters.
        valid_ranges: Optional dict from `compute_valid_ranges`; if omitted,
            only type/category checks are performed (no out-of-range warnings).

    Returns:
        A ValidationResult with any errors and warnings found.
    """
    result = ValidationResult()

    if brand not in BRAND_OPTIONS:
        result.errors.append(f"Unknown brand '{brand}'. Must be one of {BRAND_OPTIONS}.")
    if body not in BODY_OPTIONS:
        result.errors.append(f"Unknown body type '{body}'. Must be one of {BODY_OPTIONS}.")
    if engine_type not in ENGINE_TYPE_OPTIONS:
        result.errors.append(f"Unknown engine type '{engine_type}'. Must be one of {ENGINE_TYPE_OPTIONS}.")
    if registration not in REGISTRATION_OPTIONS:
        result.errors.append(f"Registration must be one of {REGISTRATION_OPTIONS}.")

    try:
        mileage = float(mileage)
        if mileage < 0:
            result.errors.append("Mileage cannot be negative.")
    except (TypeError, ValueError):
        result.errors.append("Mileage must be a number.")

    try:
        engine_v = float(engine_v)
        if engine_v <= 0:
            result.errors.append("Engine volume must be positive.")
    except (TypeError, ValueError):
        result.errors.append("Engine volume must be a number.")

    if valid_ranges and result.is_valid:
        mileage_min, mileage_max = valid_ranges["Mileage"]
        if not (mileage_min <= mileage <= mileage_max):
            result.warnings.append(
                f"Mileage {mileage:g} is outside the training data range "
                f"({mileage_min:g}-{mileage_max:g} thousand km); prediction may be less reliable."
            )
        engine_min, engine_max = valid_ranges["EngineV"]
        if not (engine_min <= engine_v <= engine_max):
            result.warnings.append(
                f"Engine volume {engine_v:g}L is outside the training data range "
                f"({engine_min:g}-{engine_max:g}L); prediction may be less reliable."
            )

    return result


def build_feature_row(
    brand: str,
    body: str,
    engine_type: str,
    registration: str,
    mileage: float,
    engine_v: float,
    feature_columns: List[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Build a single-row feature dataframe matching the model's training columns.

    Args:
        brand, body, engine_type, registration: Validated categorical inputs.
        mileage, engine_v: Validated numeric inputs.
        feature_columns: The exact column order the model/scaler expect.

    Returns:
        A one-row DataFrame with one-hot columns set and numeric columns filled.
    """
    row = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
    row["Mileage"] = mileage
    row["EngineV"] = engine_v

    for prefix, value in (("Brand", brand), ("Body", body), ("Engine Type", engine_type)):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row[col] = 1

    if registration == "yes" and "Registration_yes" in row.columns:
        row["Registration_yes"] = 1

    return row


def predict_price(
    bundle: dict,
    brand: str,
    body: str,
    engine_type: str,
    registration: str,
    mileage: float,
    engine_v: float,
) -> PredictionResult:
    """Validate inputs, run the model, and return a price estimate with an interval.

    The interval is an approximate 95% range built from the residual standard
    deviation (in log-price space) observed on the held-out test set during
    training. It is a simple empirical approximation, not a formal conformal
    or Bayesian prediction interval.

    Args:
        bundle: Model bundle from `load_model_bundle`.
        brand, body, engine_type, registration, mileage, engine_v: Raw user inputs.

    Returns:
        PredictionResult with point estimate, interval bounds, and any warnings.

    Raises:
        ValueError: If input validation fails (see `.errors` for details before
            calling this, to give the caller a chance to show a friendly message).
    """
    validation = validate_input(
        brand, body, engine_type, registration, mileage, engine_v,
        valid_ranges=bundle.get("valid_ranges"),
    )
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))

    row = build_feature_row(
        brand, body, engine_type, registration, mileage, engine_v,
        feature_columns=bundle["feature_columns"],
    )
    scaled_row = bundle["scaler"].transform(row)
    log_price_pred = bundle["model"].predict(scaled_row)[0]

    price = float(np.exp(log_price_pred))
    residual_std = bundle.get("residual_std_log", 0.0)
    price_low = float(np.exp(log_price_pred - 1.96 * residual_std))
    price_high = float(np.exp(log_price_pred + 1.96 * residual_std))

    return PredictionResult(
        price=price,
        price_low=price_low,
        price_high=price_high,
        warnings=validation.warnings,
    )
