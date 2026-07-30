"""Tests for src/data_preprocessing.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    clean_data,
    compute_valid_ranges,
    engineer_features,
    load_raw_data,
)


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_raw_data_missing_columns_raises(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"Brand": ["BMW"], "Price": [1000]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_data(bad_csv)


def test_load_raw_data_valid_file(tmp_path: Path, sample_raw_df: pd.DataFrame) -> None:
    csv_path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(csv_path, index=False)
    loaded = load_raw_data(csv_path)
    assert len(loaded) == len(sample_raw_df)


def test_clean_data_drops_missing_values(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    assert cleaned["Price"].isna().sum() == 0
    assert cleaned["Mileage"].isna().sum() == 0


def test_clean_data_removes_enginev_outliers(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    assert (cleaned["EngineV"] < 6.5).all()


def test_clean_data_removes_extreme_mileage(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    # The 5000-mileage row should be excluded by the 99th-percentile cap.
    assert cleaned["Mileage"].max() < 5000


def test_clean_data_drops_model_column(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    assert "Model" not in cleaned.columns


def test_engineer_features_returns_expected_columns(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    features, targets = engineer_features(cleaned)
    assert list(features.columns) == FEATURE_COLUMNS
    assert "Year" not in features.columns
    assert "Price" not in features.columns


def test_engineer_features_target_is_log_price(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    _, targets = engineer_features(cleaned)
    # log(Price) should be strictly positive for realistic used-car prices (>$1).
    assert (targets > 0).all()
    assert np.isclose(np.exp(targets.iloc[0]), cleaned["Price"].iloc[0], rtol=1e-6)


def test_engineer_features_no_missing_values(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    features, targets = engineer_features(cleaned)
    assert not features.isna().any().any()
    assert not targets.isna().any()


def test_compute_valid_ranges_matches_data(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(sample_raw_df)
    ranges = compute_valid_ranges(cleaned)
    assert ranges["Mileage"][0] == pytest.approx(cleaned["Mileage"].min())
    assert ranges["Mileage"][1] == pytest.approx(cleaned["Mileage"].max())
    assert ranges["EngineV"][1] <= 6.5
