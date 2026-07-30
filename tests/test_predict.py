"""Tests for src/predict.py: input validation, feature row building, and
end-to-end prediction (using the real trained model bundle when available)."""

from pathlib import Path

import pytest

from src.predict import (
    ModelLoadError,
    PredictionResult,
    build_feature_row,
    load_model_bundle,
    predict_price,
    validate_input,
)


def test_validate_input_accepts_valid_request(valid_input: dict) -> None:
    result = validate_input(**valid_input)
    assert result.is_valid
    assert result.errors == []


def test_validate_input_rejects_unknown_brand(valid_input: dict) -> None:
    valid_input["brand"] = "NotARealBrand"
    result = validate_input(**valid_input)
    assert not result.is_valid
    assert any("brand" in err.lower() for err in result.errors)


def test_validate_input_rejects_unknown_body(valid_input: dict) -> None:
    valid_input["body"] = "spaceship"
    result = validate_input(**valid_input)
    assert not result.is_valid


def test_validate_input_rejects_negative_mileage(valid_input: dict) -> None:
    valid_input["mileage"] = -50
    result = validate_input(**valid_input)
    assert not result.is_valid
    assert any("mileage" in err.lower() for err in result.errors)


def test_validate_input_rejects_non_numeric_mileage(valid_input: dict) -> None:
    valid_input["mileage"] = "a lot"
    result = validate_input(**valid_input)
    assert not result.is_valid


def test_validate_input_rejects_zero_engine_volume(valid_input: dict) -> None:
    valid_input["engine_v"] = 0
    result = validate_input(**valid_input)
    assert not result.is_valid


def test_validate_input_warns_on_out_of_range_mileage(valid_input: dict) -> None:
    valid_ranges = {"Mileage": (0, 300), "EngineV": (0.6, 6.5)}
    valid_input["mileage"] = 900
    result = validate_input(**valid_input, valid_ranges=valid_ranges)
    assert result.is_valid  # still valid, just a warning
    assert len(result.warnings) == 1


def test_build_feature_row_sets_dummy_columns(valid_input: dict) -> None:
    from src.data_preprocessing import FEATURE_COLUMNS

    row = build_feature_row(
        valid_input["brand"], valid_input["body"], valid_input["engine_type"],
        valid_input["registration"], valid_input["mileage"], valid_input["engine_v"],
        feature_columns=FEATURE_COLUMNS,
    )
    assert row.loc[0, "Mileage"] == valid_input["mileage"]
    assert row.loc[0, "EngineV"] == valid_input["engine_v"]
    assert row.loc[0, "Brand_BMW"] == 1
    assert row.loc[0, "Body_hatch"] == 0  # baseline/other categories stay zero
    assert row.loc[0, "Engine Type_Petrol"] == 1
    assert row.loc[0, "Registration_yes"] == 1


def test_build_feature_row_baseline_categories_are_all_zero() -> None:
    from src.data_preprocessing import FEATURE_COLUMNS

    # Audi / crossover / Diesel / no registration are the dropped baseline
    # categories, so every dummy column should be 0.
    row = build_feature_row("Audi", "crossover", "Diesel", "no", 100, 2.0, FEATURE_COLUMNS)
    dummy_cols = [c for c in FEATURE_COLUMNS if c not in ("Mileage", "EngineV")]
    assert (row.loc[0, dummy_cols] == 0).all()


def test_load_model_bundle_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        load_model_bundle(tmp_path / "missing.pkl")


def test_load_model_bundle_success(model_path: Path) -> None:
    if not model_path.exists():
        pytest.skip("Model bundle not trained yet; run `python -m src.train` first.")
    bundle = load_model_bundle(model_path)
    assert "model" in bundle
    assert "scaler" in bundle
    assert "feature_columns" in bundle


def test_predict_price_valid_input_returns_sane_result(model_path: Path, valid_input: dict) -> None:
    if not model_path.exists():
        pytest.skip("Model bundle not trained yet; run `python -m src.train` first.")
    bundle = load_model_bundle(model_path)
    result = predict_price(bundle, **valid_input)
    assert isinstance(result, PredictionResult)
    assert result.price > 0
    assert result.price_low <= result.price <= result.price_high


def test_predict_price_invalid_input_raises(model_path: Path, valid_input: dict) -> None:
    if not model_path.exists():
        pytest.skip("Model bundle not trained yet; run `python -m src.train` first.")
    bundle = load_model_bundle(model_path)
    valid_input["brand"] = "NotARealBrand"
    with pytest.raises(ValueError):
        predict_price(bundle, **valid_input)
