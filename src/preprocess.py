"""
preprocess.py
=============
Data cleaning + feature engineering for ResaleRadar.

Responsibilities
----------------
* Load used_device_data.csv
* Clean column names / value formats and engineer ``device_age``
* Handle missing values (median for numeric, mode for categorical)
* Encode categorical columns with LabelEncoder (with an "unknown" class so
  unseen categories at prediction time never crash the app)
* Scale numeric features with StandardScaler
* Save the fitted scaler  -> models/scaler.pkl
* Save the fitted encoders -> models/encoders.pkl
* Save helper metadata (fill values, price tiers, ...) -> models/preprocess_meta.pkl

Run on its own (from the project root) for a quick sanity check:
    python src/preprocess.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "used_device_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
META_PATH = MODELS_DIR / "preprocess_meta.pkl"

TARGET_COLUMN = "normalized_used_price"

# The dataset was collected around 2021; device age = REFERENCE_YEAR - release_year.
REFERENCE_YEAR = 2021

# Label used for categories that were not seen during training.
UNKNOWN_TOKEN = "__unknown__"

# The raw columns "4g" and "5g" start with a digit, so we rename them.
COLUMN_RENAMES = {"4g": "four_g", "5g": "five_g"}

CATEGORICAL_COLUMNS = ["device_brand", "os", "four_g", "five_g"]
NUMERIC_COLUMNS = [
    "screen_size",           # cm in the dataset
    "main_camera_mp",        # rear camera
    "selfie_camera_mp",      # front camera
    "int_memory",            # storage in GB
    "ram",                   # GB
    "battery",               # mAh
    "weight",                # grams
    "days_used",
    "normalized_new_price",
    "device_age",            # engineered: REFERENCE_YEAR - release_year
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_data(path=DATA_PATH):
    """Read the raw CSV file and return it as a DataFrame (clear error if missing)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download 'used_device_data.csv' from Kaggle (ahsan81) and place it "
            "inside the data/ folder."
        )
    return pd.read_csv(path)


def clean_data(df, require_target=True):
    """
    Standardise the raw DataFrame:
    - strip/rename columns, engineer device_age from release_year
    - normalise text values (strip spaces, lower-case yes/no flags)
    - coerce numeric columns to numbers (bad values become NaN)
    - drop rows whose target is missing
    Returns a new DataFrame containing FEATURE_COLUMNS (+ target if present).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_RENAMES)

    # Feature engineering: age of the device in years at the reference year.
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df["device_age"] = REFERENCE_YEAR - df["release_year"]
        df = df.drop(columns=["release_year"])

    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    # Categorical columns: strip whitespace, empty strings become NaN.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].apply(lambda v: np.nan if pd.isna(v) else str(v).strip())
        df[col] = df[col].replace("", np.nan)
    for col in ("four_g", "five_g"):
        df[col] = df[col].apply(lambda v: v if pd.isna(v) else v.lower())

    # Numeric columns: anything non-numeric becomes NaN (filled later).
    numeric_cols = NUMERIC_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows without a target cannot be used for training.
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    keep = FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def compute_fill_values(df):
    """Learn the fill value for every feature: median (numeric) / mode (categorical)."""
    numeric = {}
    for col in NUMERIC_COLUMNS:
        median = df[col].median()
        numeric[col] = 0.0 if pd.isna(median) else float(median)

    categorical = {}
    for col in CATEGORICAL_COLUMNS:
        modes = df[col].mode()
        categorical[col] = modes.iloc[0] if len(modes) else UNKNOWN_TOKEN
    return {"numeric": numeric, "categorical": categorical}


def fill_missing(df, fill_values):
    """Fill NaNs using the previously learned medians / modes. Returns a new DataFrame."""
    df = df.copy()
    for col, value in fill_values["numeric"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    for col, value in fill_values["categorical"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


# --------------------------------------------------------------------------- #
# Encoding and scaling
# --------------------------------------------------------------------------- #
def fit_encoders(df):
    """
    Fit one LabelEncoder per categorical column.
    An extra UNKNOWN_TOKEN class is added so unseen values can be encoded later.
    """
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        classes = sorted(df[col].astype(str).unique()) + [UNKNOWN_TOKEN]
        encoder = LabelEncoder()
        encoder.fit(classes)
        encoders[col] = encoder
    return encoders


def apply_encoders(df, encoders):
    """
    Encode categorical columns with the fitted encoders.
    Matching is case-insensitive; unseen categories map to UNKNOWN_TOKEN.
    """
    out = df.copy()
    for col, encoder in encoders.items():
        lookup = {str(c).lower(): c for c in encoder.classes_}
        cleaned = out[col].astype(str).str.strip().str.lower()
        mapped = cleaned.map(lookup).fillna(UNKNOWN_TOKEN)
        out[col] = encoder.transform(mapped)
    return out


def transform_features(df, scaler, encoders, meta):
    """
    Turn a cleaned (raw-valued) DataFrame into the model-ready feature matrix:
    fill missing -> encode categoricals -> scale numerics -> order columns.
    Works both for the training data and for a single listing at prediction time.
    """
    data = df.reindex(columns=FEATURE_COLUMNS).copy()
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = fill_missing(data, meta["fill_values"])
    data = apply_encoders(data, encoders)
    data[NUMERIC_COLUMNS] = scaler.transform(data[NUMERIC_COLUMNS])
    return data[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# Fit + save / load artifacts
# --------------------------------------------------------------------------- #
def fit_preprocessor(df_clean, save=True):
    """
    Fit the whole preprocessing pipeline on a cleaned DataFrame.

    Returns (X, y, artifacts) where artifacts = {"scaler", "encoders", "meta"}.
    When ``save`` is True the scaler, encoders and metadata are written to models/.
    """
    fill_values = compute_fill_values(df_clean)
    filled = fill_missing(df_clean, fill_values)

    encoders = fit_encoders(filled)
    encoded = apply_encoders(filled, encoders)
    scaler = StandardScaler().fit(encoded[NUMERIC_COLUMNS])

    target = df_clean[TARGET_COLUMN]
    meta = {
        "fill_values": fill_values,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "reference_year": REFERENCE_YEAR,
        # Used by the app to label predictions Budget / Mid-range / Premium.
        "price_tiers": (float(target.quantile(1 / 3)), float(target.quantile(2 / 3))),
        "target_median": float(target.median()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }

    X = transform_features(df_clean, scaler, encoders, meta)
    y = target.reset_index(drop=True)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(meta, META_PATH)

    return X, y, {"scaler": scaler, "encoders": encoders, "meta": meta}


def load_artifacts():
    """Load (scaler, encoders, meta) from models/. Raises a helpful error if missing."""
    for path in (SCALER_PATH, ENCODERS_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing '{path.name}'. Run `python src/train_model.py` first."
            )
    return joblib.load(SCALER_PATH), joblib.load(ENCODERS_PATH), joblib.load(META_PATH)


# --------------------------------------------------------------------------- #
# Stand-alone sanity check
# --------------------------------------------------------------------------- #
def run_preprocessing():
    """Load, clean and fit the preprocessing pipeline, saving all artifacts."""
    print("Loading dataset...")
    raw = load_data()
    print(f"  Raw shape: {raw.shape}")

    df = clean_data(raw)
    print(f"  Cleaned shape: {df.shape}")
    print(f"  Missing values before filling:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    X, y, _ = fit_preprocessor(df, save=True)
    print(f"Feature matrix: {X.shape}, target: {y.shape}")
    print(f"Saved scaler, encoders and metadata to '{MODELS_DIR}'.")
    return X, y


if __name__ == "__main__":
    run_preprocessing()