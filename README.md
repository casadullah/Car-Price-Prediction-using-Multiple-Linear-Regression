# Car Price Predictor

A machine learning app that estimates a used car's market price from its
brand, body type, mileage, engine, registration status, and engine type.
Built as a portfolio piece to demonstrate an end-to-end ML workflow: data
cleaning, model comparison, hyperparameter tuning, testing, and deployment
as an interactive Streamlit app.

**Business use case:** a quick, defensible starting-point valuation for a
used-car listing, trade-in estimate, or buyer's sanity check — the kind of
"what should this be worth" question a dealer, marketplace, or private
seller asks before negotiating.

---

## Live app

Run locally with `streamlit run streamlit_app.py` (see [How to Run](#how-to-run)).

---

## Methodology

### Data

Source: `data/Car_Sales_Raw_Data.csv` — 4,345 used car listings scraped from
a European classifieds site, covering 7 brands (Audi, BMW, Mercedes-Benz,
Mitsubishi, Renault, Toyota, Volkswagen).

Cleaning steps (see `src/data_preprocessing.py`):
1. Drop rows with missing values (320 rows dropped).
2. Cap outliers: Mileage at its 99th percentile, EngineV at 6.5L, Year at its
   1st percentile.
3. Log-transform `Price` — used-car prices are heavily right-skewed; modeling
   `log(Price)` gives a much more linear, homoscedastic target.
4. One-hot encode `Brand`, `Body`, `Engine Type`, `Registration`
   (`drop_first=True` to avoid the dummy variable trap).
5. Drop `Year` — a Variance Inflation Factor (VIF) analysis showed it was
   highly collinear with Mileage and EngineV (VIF ≈ 10.3), so it was removed
   from the feature set rather than double-counting the same signal.

Result: 3,907 cleaned rows, 17 features, log-price target.

### Model comparison

Three model families were trained and compared using 5-fold cross-validation
on a 70% training split (validated on a further 15%, tested on the final
held-out 15%):

| Model | 5-fold CV R² (log-price, train set) |
|---|---|
| Linear Regression (baseline) | 0.7459 ± 0.0175 |
| Random Forest | 0.7756 ± 0.0163 |
| **Gradient Boosting** | **0.7833 ± 0.0144** |

**Gradient Boosting was selected** for the best mean CV R² and lowest
variance across folds, then tuned with `RandomizedSearchCV` (20 iterations,
5-fold) over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and
`min_samples_leaf`. Tuning raised CV R² to **0.7973**.

> **Note on algorithm choice:** the brief called for XGBoost or LightGBM.
> This machine's Python (3.14, bleeding-edge) has no prebuilt wheels for
> either package, and building them from source hung for 25+ minutes with
> no clear ETA. scikit-learn's `GradientBoostingRegressor` was used instead
> — same boosted-trees family, no compiled-dependency risk. Swapping in
> XGBoost/LightGBM later is a drop-in change to `src/train.py`'s
> `CANDIDATE_MODELS` dict.

### Why not just trust the R²?

A single train/test R² can be misleading on a dataset this size (~4k rows)
with a heavy-tailed price distribution (a few $100k+ luxury cars sit next to
a mass of sub-$20k listings). Reporting 5-fold CV mean **and** standard
deviation, plus RMSE/MAE/MAPE on a truly held-out test set, gives a more
honest picture of how much the "best" model's score could swing on a
different sample.

---

## Performance metrics

### Final model: tuned Gradient Boosting

| Split | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Validation (15%) | 0.852 | $10,011 | $4,998 | 26.8% |
| **Test (15%, held-out)** | **0.675** | **$15,920** | **$6,054** | **36.5%** |

*(R² is also reported at 0.797 on the log-price scale for the test set,
which is the scale the model is actually optimized on — the price-scale
numbers above are what a user actually experiences in dollars.)*

### Old vs. new: was it worth it?

To answer honestly, the original single Linear Regression model was
re-evaluated with the exact same metrics used for the new model (same data,
different split methodology — old used an 80/20 split, new uses 70/15/15):

| Metric (test set, price scale) | Old (Linear Regression) | New (tuned Gradient Boosting) | Change |
|---|---|---|---|
| R² | 0.621 | 0.675 | +0.054 |
| RMSE | $15,775 | $15,920 | +$145 (worse) |
| MAE | $6,674 | $6,054 | −$620 (better) |
| MAPE | 35.0% | 36.5% | +1.5 pp (worse) |
| R² (log-price scale) | 0.783 | 0.797 | +0.014 |

**Honest takeaway:** the upgrade is a modest, not dramatic, improvement.
R² and MAE both improved meaningfully, but RMSE and MAPE are essentially
flat to slightly worse — a handful of high-price outliers in the small
(587-row) test set dominate both of those metrics and swing them run-to-run.
On a dataset this size, don't expect ensemble methods to work miracles over
a well-specified linear model; the real gains here are in **robustness**
(cross-validated model selection instead of a single lucky split),
**honesty** (four metrics instead of one, reported on a true held-out set),
and **production readiness** (validation, testing, logging), not raw
accuracy.

### Feature importance

What the model actually weighs (Gradient Boosting's built-in
`feature_importances_`):

| Feature | Importance |
|---|---|
| Mileage | 46.4% |
| Engine Volume | 28.7% |
| Registration (yes) | 14.4% |
| Brand: Mercedes-Benz | 1.5% |
| Engine Type: Petrol | 1.5% |
| Brand: Mitsubishi | 1.4% |
| Brand: Renault | 1.3% |
| Body: sedan | 1.1% |

Mileage, engine size, and registration status drive ~90% of the price
signal; brand and body type matter, but far less than these three.

---

## Model limitations

Be honest about what this model can and can't do:

- **Small dataset.** ~3,900 training rows after cleaning is small for a
  gradient-boosted ensemble; expect meaningful variance between retrains
  with different random seeds.
- **Narrow brand coverage.** Only 7 brands (Audi, BMW, Mercedes-Benz,
  Mitsubishi, Renault, Toyota, Volkswagen). A Ford, Honda, or Tesla will be
  silently unsupported — the app's dropdowns only offer trained categories.
- **Valid input ranges** (extrapolation outside these is flagged as a
  warning but not blocked): Mileage 0–433 thousand km, Engine Volume
  0.6–6.3L, Year 1988–2016 (Year is documented but not a model input).
- **Heavy-tailed test error.** MAPE of ~36% on the test set means a
  meaningful fraction of predictions miss by more than a third of true
  price, concentrated in high-end luxury cars, which are underrepresented
  in the training data.
- **Prediction intervals are approximate.** The ±95% range shown in the app
  is built from the held-out test set's residual standard deviation in
  log-price space — a simple empirical approximation, not a formal
  conformal or Bayesian interval. Treat it as "roughly how wrong this model
  tends to be," not a calibrated statistical guarantee.
- **Market data is dated.** The dataset covers listings only through 2016;
  it reflects a used-car market from that period and geography, not current
  prices or inflation.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │  Car_Sales_Raw_Data.csv │
                    └────────────┬────────────┘
                                 │
                     src/data_preprocessing.py
                (clean → log-transform → one-hot encode)
                                 │
                                 ▼
                         src/train.py
        ┌──────────────┬──────────────┬──────────────┐
        │ LinearReg    │ RandomForest │ GradientBoost│  ← 5-fold CV
        └──────────────┴──────────────┴──────┬───────┘
                                              │ (best candidate)
                                   RandomizedSearchCV tuning
                                              │
                                              ▼
                              model/car_price_model.pkl
                   (model + scaler + metrics + valid ranges
                    + feature importance, via joblib)
                                              │
                    ┌─────────────────────────┼─────────────────────┐
                    ▼                                                ▼
             src/predict.py                                streamlit_app.py
     (validate input → build features →                (form → predict.py →
        scale → predict → interval)                    show price + range +
                    ▲                                    performance + charts)
                    │
              tests/ (pytest)
     validates preprocessing, model loading,
        valid/invalid prediction handling
```

---

## Project structure

```
├── data/
│   └── Car_Sales_Raw_Data.csv       # Raw dataset
├── model/
│   └── car_price_model.pkl          # Trained model bundle (joblib)
├── notebook/
│   └── Practical_Example.ipynb      # Original exploratory notebook
├── src/
│   ├── data_preprocessing.py        # Cleaning, encoding, feature engineering
│   ├── train.py                     # Model comparison, tuning, saving
│   ├── predict.py                   # Input validation + prediction
│   └── evaluate.py                  # Metrics, CV summaries, feature importance
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_predict.py
│   └── test_evaluate.py
├── streamlit_app.py                 # Streamlit UI
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # + pytest, for running tests
└── runtime.txt                      # Pins Python version for Streamlit Cloud
```

---

## How to run

### Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (writes model/car_price_model.pkl)
python -m src.train

# 3. Launch the app
streamlit run streamlit_app.py
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Future improvements

- Swap in XGBoost/LightGBM once running on a Python version with prebuilt
  wheels for them, and re-run the same comparison harness in `src/train.py`.
- Expand the dataset with more recent listings and broader brand/model
  coverage to reduce the luxury-car blind spot called out in Limitations.
- Replace the empirical residual-based interval with a proper conformal
  prediction interval for calibrated uncertainty estimates.
- Add SHAP values for per-prediction explanations (which factors pushed
  *this specific* car's price up or down), not just global feature
  importance.
- Track retraining runs (MLflow or similar) so metric drift across dataset
  updates is visible over time instead of only at the moment of retraining.
