"""Metrics computation, cross-validation summaries, and feature importance."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE as a percentage.

    Args:
        y_true: Ground-truth values (original price scale, not log scale).
        y_pred: Predicted values (original price scale).

    Returns:
        Mean absolute percentage error, in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R^2, RMSE, MAE, and MAPE on the original (non-log) price scale.

    Args:
        y_true: Ground-truth prices.
        y_pred: Predicted prices.

    Returns:
        Dict with keys "r2", "rmse", "mae", "mape".
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def cross_validate_model(
    model: BaseEstimator,
    features: pd.DataFrame,
    log_targets: pd.Series,
    n_splits: int = 5,
    random_state: int = 365,
) -> Dict[str, Any]:
    """Run k-fold cross-validation and summarize R^2 (log-price scale).

    Cross-validation is run in log-price space (the model's native training
    target) since that is what `cross_validate` scores directly; RMSE/MAE/MAPE
    on the original price scale are reported separately from the held-out
    test set in `train.py`.

    Args:
        model: An unfitted scikit-learn regressor.
        features: Feature matrix.
        log_targets: log(Price) targets.
        n_splits: Number of CV folds.
        random_state: Seed for fold shuffling.

    Returns:
        Dict with "r2_mean", "r2_std", and the raw per-fold "r2_scores".
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(model, features, log_targets, cv=kfold, scoring="r2")
    scores = results["test_score"]
    logger.info(
        "%d-fold CV for %s: R^2 mean=%.4f std=%.4f",
        n_splits, type(model).__name__, scores.mean(), scores.std(),
    )
    return {
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std()),
        "r2_scores": scores.tolist(),
    }


def get_feature_importance(model: BaseEstimator, feature_names: list) -> pd.DataFrame:
    """Extract feature importance (tree-based) or absolute coefficients (linear).

    Args:
        model: A fitted scikit-learn regressor exposing either
            `feature_importances_` or `coef_`.
        feature_names: Column names matching the model's input features.

    Returns:
        DataFrame with columns ["Feature", "Importance"], sorted descending.
        Empty DataFrame if the model exposes neither attribute.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model %s exposes no importance/coef attribute", type(model).__name__)
        return pd.DataFrame(columns=["Feature", "Importance"])

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)
