"""Model training pipeline: compare Linear Regression, Random Forest, and
Gradient Boosting; tune the best candidate; and save the winner.

Usage:
    python -m src.train
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_and_prepare
from src.evaluate import compute_metrics, cross_validate_model, get_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Car_Sales_Raw_Data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "car_price_model.pkl"

RANDOM_STATE = 365

CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

# RandomizedSearchCV param distributions, keyed by candidate name.
PARAM_DISTRIBUTIONS = {
    "RandomForest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.85, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
}


def split_data(
    features: pd.DataFrame, targets: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into 70% train / 15% validation / 15% test.

    Args:
        features: Full feature matrix.
        targets: Full log-price target vector.

    Returns:
        (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, targets, test_size=0.30, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(x_train), len(x_val), len(x_test)
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def compare_candidates(
    x_train_scaled: np.ndarray, y_train: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Run 5-fold CV for each candidate model and collect summary stats.

    Args:
        x_train_scaled: Scaled training features.
        y_train: log-price training targets.

    Returns:
        Dict keyed by model name -> CV summary dict (see `cross_validate_model`).
    """
    results = {}
    for name, model in CANDIDATE_MODELS.items():
        logger.info("Running 5-fold CV for %s", name)
        results[name] = cross_validate_model(model, x_train_scaled, y_train, n_splits=5)
    return results


def tune_best_candidate(
    best_name: str, x_train_scaled: np.ndarray, y_train: pd.Series
) -> Any:
    """Hyperparameter-tune the best-performing candidate via RandomizedSearchCV.

    Falls back to the untuned estimator if `best_name` has no defined search
    space (e.g. LinearRegression, which has no hyperparameters worth tuning here).

    Args:
        best_name: Key into CANDIDATE_MODELS / PARAM_DISTRIBUTIONS.
        x_train_scaled: Scaled training features.
        y_train: log-price training targets.

    Returns:
        A fitted, tuned estimator.
    """
    base_model = CANDIDATE_MODELS[best_name]
    if best_name not in PARAM_DISTRIBUTIONS:
        logger.info("No tuning search space for %s; fitting as-is", best_name)
        base_model.fit(x_train_scaled, y_train)
        return base_model

    search = RandomizedSearchCV(
        base_model,
        PARAM_DISTRIBUTIONS[best_name],
        n_iter=20,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logger.info("Running RandomizedSearchCV for %s (20 iters, 5-fold)", best_name)
    search.fit(x_train_scaled, y_train)
    logger.info("Best params for %s: %s", best_name, search.best_params_)
    logger.info("Best CV R^2 for %s: %.4f", best_name, search.best_score_)
    return search.best_estimator_


def main() -> None:
    """Train, compare, tune, and save the best car price model."""
    try:
        features, log_targets, valid_ranges = load_and_prepare(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load/prepare data: %s", exc)
        raise

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(features, log_targets)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    cv_results = compare_candidates(x_train_scaled, y_train)

    # Select the candidate with the best mean CV R^2 for tuning.
    best_name = max(cv_results, key=lambda name: cv_results[name]["r2_mean"])
    logger.info("Best candidate by CV R^2: %s", best_name)

    tuned_model = tune_best_candidate(best_name, x_train_scaled, y_train)

    # Confirm on the validation set (guards against a tuned model overfitting CV folds).
    val_pred_log = tuned_model.predict(x_val_scaled)
    val_metrics = compute_metrics(np.exp(y_val), np.exp(val_pred_log))
    logger.info("Validation metrics (%s): %s", best_name, val_metrics)

    # Refit on train+val before final test evaluation, to use all non-test data.
    x_trainval_scaled = np.vstack([x_train_scaled, x_val_scaled])
    y_trainval = pd.concat([y_train, y_val])
    tuned_model.fit(x_trainval_scaled, y_trainval)

    test_pred_log = tuned_model.predict(x_test_scaled)
    test_metrics = compute_metrics(np.exp(y_test), np.exp(test_pred_log))
    logger.info("Held-out test metrics (%s): %s", best_name, test_metrics)

    residual_std_log = float(np.std(y_test.values - test_pred_log))

    feature_importance = get_feature_importance(tuned_model, list(features.columns))

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": tuned_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "brands": ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"],
        "bodies": ["crossover", "hatch", "other", "sedan", "vagon", "van"],
        "engine_types": ["Diesel", "Gas", "Other", "Petrol"],
        "registrations": ["yes", "no"],
        "valid_ranges": valid_ranges,
        "cv_results": cv_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "residual_std_log": residual_std_log,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }
    try:
        joblib.dump(bundle, MODEL_PATH)
    except OSError as exc:
        logger.error("Failed to save model bundle to %s: %s", MODEL_PATH, exc)
        raise
    logger.info("Saved model bundle (%s) to %s", best_name, MODEL_PATH)

    print("\n=== Cross-validation summary (train set, 5-fold, R^2 on log-price) ===")
    for name, res in cv_results.items():
        print(f"{name:>18}: R^2 = {res['r2_mean']:.4f} +/- {res['r2_std']:.4f}")

    print(f"\n=== Selected model: {best_name} ===")
    print("Validation metrics (price scale):", val_metrics)
    print("Test metrics (price scale):", test_metrics)
    print("\nTop feature importances:")
    print(feature_importance.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
