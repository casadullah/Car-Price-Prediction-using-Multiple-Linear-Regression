"""Tests for src/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluate import (
    compute_metrics,
    cross_validate_model,
    get_feature_importance,
    mean_absolute_percentage_error,
)


def test_mean_absolute_percentage_error_perfect_prediction() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    assert mean_absolute_percentage_error(y_true, y_true) == 0.0


def test_mean_absolute_percentage_error_known_value() -> None:
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)


def test_compute_metrics_returns_expected_keys() -> None:
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"r2", "rmse", "mae", "mape"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["mape"] >= 0


def test_compute_metrics_perfect_fit_gives_r2_of_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y_true, y_true)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_cross_validate_model_returns_five_scores() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = x["a"] * 2 + x["b"] - x["c"] + rng.normal(scale=0.01, size=50)
    result = cross_validate_model(LinearRegression(), x, y, n_splits=5)
    assert len(result["r2_scores"]) == 5
    assert result["r2_mean"] > 0.9  # near-linear synthetic data should fit well


def test_get_feature_importance_linear_model_uses_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, 2)), columns=["strong", "weak"])
    y = x["strong"] * 10 + x["weak"] * 0.01
    model = LinearRegression().fit(x, y)
    importance = get_feature_importance(model, ["strong", "weak"])
    assert importance.iloc[0]["Feature"] == "strong"


def test_get_feature_importance_handles_model_without_attributes() -> None:
    class Dummy:
        pass

    importance = get_feature_importance(Dummy(), ["a", "b"])
    assert importance.empty
