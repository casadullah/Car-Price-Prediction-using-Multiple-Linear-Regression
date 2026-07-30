"""Shared pytest fixtures for the car price prediction test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small synthetic raw dataframe covering the cleaning edge cases:
    one row with a missing value, one EngineV outlier, one Year outlier,
    and several normal rows.
    """
    rng = np.random.default_rng(42)
    n_normal = 30
    normal_rows = pd.DataFrame(
        {
            "Brand": rng.choice(
                ["Audi", "BMW", "Mercedes-Benz", "Toyota", "Volkswagen"], n_normal
            ),
            "Price": rng.uniform(2000, 40000, n_normal),
            "Body": rng.choice(["sedan", "hatch", "crossover", "van"], n_normal),
            "Mileage": rng.uniform(0, 300, n_normal),
            "EngineV": rng.uniform(1.0, 4.0, n_normal),
            "Engine Type": rng.choice(["Diesel", "Petrol", "Gas"], n_normal),
            "Registration": rng.choice(["yes", "no"], n_normal),
            "Year": rng.integers(2000, 2016, n_normal),
            "Model": "SomeModel",
        }
    )

    edge_rows = pd.DataFrame(
        [
            {  # missing Price -> should be dropped by dropna
                "Brand": "BMW", "Price": np.nan, "Body": "sedan", "Mileage": 100,
                "EngineV": 2.0, "Engine Type": "Petrol", "Registration": "yes",
                "Year": 2010, "Model": "X",
            },
            {  # EngineV outlier -> should be capped out (>= 6.5)
                "Brand": "Audi", "Price": 15000, "Body": "sedan", "Mileage": 100,
                "EngineV": 9.9, "Engine Type": "Gas", "Registration": "yes",
                "Year": 2010, "Model": "X",
            },
            {  # extreme mileage outlier -> should be filtered by the 99th pct cap
                "Brand": "Toyota", "Price": 5000, "Body": "van", "Mileage": 5000,
                "EngineV": 2.0, "Engine Type": "Diesel", "Registration": "no",
                "Year": 2005, "Model": "X",
            },
        ]
    )

    return pd.concat([normal_rows, edge_rows], ignore_index=True)


@pytest.fixture
def model_path() -> Path:
    """Path to the trained model bundle (may not exist until `src.train` runs)."""
    return BASE_DIR / "model" / "car_price_model.pkl"


@pytest.fixture
def valid_input() -> dict:
    """A known-good set of prediction inputs within the training distribution."""
    return {
        "brand": "BMW",
        "body": "sedan",
        "engine_type": "Petrol",
        "registration": "yes",
        "mileage": 150,
        "engine_v": 2.0,
    }
