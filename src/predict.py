"""
predict.py
==========
Single-prediction utility used by the Streamlit app (and usable from the
command line / notebooks).

Main entry point:
    predict_price(user_input: dict, model_name: str) -> dict

Expected keys in ``user_input`` (all optional - missing values fall back to the
training median / mode and a note is returned):

    device_brand, os, four_g ("yes"/"no"), five_g ("yes"/"no"),
    screen_size_inches  (or screen_size in cm), main_camera_mp (rear),
    selfie_camera_mp (front), ram (GB), battery (mAh), int_memory (storage GB),
    device_age (years since launch),
    and optionally: weight (g), days_used, normalized_new_price
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import (
    CATEGORICAL_COLUMNS,
    META_PATH,
    MODELS_DIR,
    NUMERIC_COLUMNS,
    SCALER_PATH,
    ENCODERS_PATH,
    UNKNOWN_TOKEN,
    transform_features,
)

INCH_TO_CM = 2.54  # the dataset stores screen size in centimetres
PRICE_BAND = 0.10  # +/- 10 % price band

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# Columns that must be strictly positive to make physical sense.
POSITIVE_ONLY = {"screen_size", "ram", "battery", "int_memory"}

FRIENDLY_NAMES = {
    "device_brand": "brand", "os": "OS", "four_g": "4G", "five_g": "5G",
    "screen_size": "screen size", "main_camera_mp": "rear camera",
    "selfie_camera_mp": "front camera", "int_memory": "storage", "ram": "RAM",
    "battery": "battery", "weight": "weight", "days_used": "days used",
    "normalized_new_price": "new price", "device_age": "device age",
}

# Simple cache so pickles are not re-read on every Streamlit interaction.
_CACHE = {}


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def cached_load(path):
    """Load a pickle with joblib, caching by (path, modification time)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file '{path.name}'. Run the training scripts first "
            "(python src/train_model.py, then python src/fraud_detection.py)."
        )
    key = (str(path), path.stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def get_preprocessing_artifacts():
    """Return (scaler, encoders, meta) loaded from models/."""
    return cached_load(SCALER_PATH), cached_load(ENCODERS_PATH), cached_load(META_PATH)


def load_model(model_name):
    """Load one of the three trained regressors by its display name."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_FILES)}.")
    return cached_load(MODELS_DIR / MODEL_FILES[model_name])


# --------------------------------------------------------------------------- #
# Input cleaning
# --------------------------------------------------------------------------- #
def to_number(value):
    """Convert a value to a finite, non-negative float; return None if impossible."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0:
        return None
    return number


def _to_yes_no(value):
    """Normalise yes/no style inputs (bool, 1/0, 'Yes', 'y', ...) to 'yes'/'no' or None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "1.0"}:
        return "yes"
    if text in {"no", "n", "false", "0", "0.0"}:
        return "no"
    return None


def prepare_raw_row(user_input):
    """
    Convert a (possibly incomplete / messy) user_input dict into a one-row
    DataFrame with raw feature values.
    Missing, invalid or negative values are replaced by training medians / modes.
    Returns (row_df, notes) where notes is a list of human-readable warnings.
    """
    _, encoders, meta = get_preprocessing_artifacts()
    fills = meta["fill_values"]
    data = dict(user_input or {})
    row, defaulted, notes = {}, [], []

    # Numeric features (screen size may arrive in inches -> convert to cm).
    numeric_values = {}
    inches = to_number(data.get("screen_size_inches"))
    numeric_values["screen_size"] = (
        inches * INCH_TO_CM if inches else to_number(data.get("screen_size"))
    )
    for col in NUMERIC_COLUMNS:
        if col != "screen_size":
            numeric_values[col] = to_number(data.get(col))

    for col in NUMERIC_COLUMNS:
        value = numeric_values[col]
        if value is not None and col in POSITIVE_ONLY and value <= 0:
            value = None
        if value is None:
            value = fills["numeric"][col]
            defaulted.append(FRIENDLY_NAMES[col])
        row[col] = float(value)

    # Categorical features.
    for col in CATEGORICAL_COLUMNS:
        if col in ("four_g", "five_g"):
            value = _to_yes_no(data.get(col))
        else:
            raw = data.get(col)
            value = None if raw is None or not str(raw).strip() else str(raw).strip()
        if value is None:
            value = fills["categorical"][col]
            defaulted.append(FRIENDLY_NAMES[col])
        row[col] = value

        # Warn when a brand / OS was never seen in training.
        known = {str(c).lower() for c in encoders[col].classes_}
        if str(value).lower() not in known or str(value) == UNKNOWN_TOKEN:
            notes.append(
                f"{FRIENDLY_NAMES[col].capitalize()} '{value}' was not seen during "
                "training and is treated as an unknown category."
            )

    if defaulted:
        notes.insert(
            0,
            "Not provided / invalid - dataset median or most common value used for: "
            + ", ".join(defaulted) + ".",
        )
    return pd.DataFrame([row]), notes


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_from_raw_row(raw_row, model_name="Random Forest"):
    """Run preprocessing + the selected model on a one-row raw DataFrame; return a float."""
    scaler, encoders, meta = get_preprocessing_artifacts()
    features = transform_features(raw_row, scaler, encoders, meta)
    model = load_model(model_name)
    return float(model.predict(features)[0])


def price_segment(price, price_tiers):
    """Map a predicted normalized price to Budget / Mid-range / Premium using training tertiles."""
    low, high = price_tiers
    if price < low:
        return "Budget"
    if price < high:
        return "Mid-range"
    return "Premium"


def predict_price(user_input, model_name="Random Forest"):
    """
    Predict the fair (normalized) used price for one device.

    Returns a dict with: model, predicted_price, lower, upper (+/-10 % band),
    segment (Budget / Mid-range / Premium), price_tiers and notes (warnings).
    """
    raw_row, notes = prepare_raw_row(user_input)
    predicted = predict_from_raw_row(raw_row, model_name)
    _, _, meta = get_preprocessing_artifacts()
    return {
        "model": model_name,
        "predicted_price": predicted,
        "lower": predicted * (1 - PRICE_BAND),
        "upper": predicted * (1 + PRICE_BAND),
        "segment": price_segment(predicted, meta["price_tiers"]),
        "price_tiers": meta["price_tiers"],
        "notes": notes,
    }


if __name__ == "__main__":
    # Quick command-line demo with an example phone.
    example = {
        "device_brand": "Samsung", "os": "Android", "screen_size_inches": 6.1,
        "four_g": "yes", "five_g": "no", "selfie_camera_mp": 8, "main_camera_mp": 13,
        "ram": 4, "battery": 4000, "int_memory": 64, "device_age": 2,
    }
    for name in MODEL_FILES:
        result = predict_price(example, name)
        print(f"{name:18s} -> {result['predicted_price']:.3f} "
              f"[{result['lower']:.3f}, {result['upper']:.3f}] {result['segment']}")