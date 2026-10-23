"""
train_model.py
==============
Trains the three price-estimation models used by ResaleRadar:

    1. Linear Regression     -> models/linear_regression.pkl
    2. Random Forest         -> models/random_forest.pkl
    3. XGBoost               -> models/xgboost.pkl

For every model MAE, RMSE and R2 are printed and saved together in
models/model_comparison.csv.

Run from the project root:
    python src/train_model.py
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import (
    MODELS_DIR,
    TARGET_COLUMN,
    transform_features,
    clean_data,
    fit_preprocessor,
    load_data,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_model(y_true, y_pred):
    """Return a dict with MAE, RMSE and R2 for the given predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_models():
    """Create the three (untrained) regressors with sensible laptop-friendly settings."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - only hit when xgboost is missing
        raise ImportError("xgboost is not installed. Run: pip install -r requirements.txt") from exc

    return {
        "Linear Regression": (
            LinearRegression(),
            "linear_regression.pkl",
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=1,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "random_forest.pkl",
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "xgboost.pkl",
        ),
    }


def train_all_models():
    """
    Full training pipeline:
    load -> clean -> split -> fit preprocessing on the TRAIN split only ->
    train + evaluate 3 models -> save models and the comparison table.
    Returns the comparison DataFrame.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df = clean_data(load_data())
    print(f"  {len(df)} rows, {df.shape[1] - 1} feature columns.")

    # Split BEFORE fitting scaler/encoders so test data never leaks into them.
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Fitting preprocessing pipeline (imputer, encoders, scaler)...", end=" ", flush=True)
    X_train, y_train, artifacts = fit_preprocessor(train_df, save=True)
    X_test = transform_features(
        test_df, artifacts["scaler"], artifacts["encoders"], artifacts["meta"]
    )
    y_test = test_df[TARGET_COLUMN]
    print("Done.")

    results = []
    for name, (model, filename) in build_models().items():
        print(f"Training {name}...", end=" ", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print("Done.")

        metrics = evaluate_model(y_test, model.predict(X_test))
        joblib.dump(model, MODELS_DIR / filename)
        print(
            f"  {name}: MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
            f"R2={metrics['R2']:.4f}  (trained in {elapsed:.1f}s)  -> models/{filename}"
        )
        results.append({"Model": name, **metrics, "Train_Time_Sec": round(elapsed, 2)})

    comparison = pd.DataFrame(results)
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    print("\n=== Model comparison (test set) ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSaved comparison table to {MODELS_DIR / 'model_comparison.csv'}")
    return comparison


if __name__ == "__main__":
    train_all_models()