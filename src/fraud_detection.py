"""
fraud_detection.py
==================
Unsupervised fraud / scam-listing detection with Isolation Forest.

Idea
----
A listing is suspicious when its asking price is unusually LOW compared with
what its specifications are normally worth.  To capture that, every listing is
described by:

    price_gap  = listed price - fair price (fair price comes from a Random Forest
                 that sees ALL the specs; training-set estimates are out-of-fold,
                 5-fold CV, so the gap is honest)
    plus the listed price itself.

The specs therefore enter the detector through the fair-price estimate.  Keeping
the Isolation Forest focused on these two signals stops the price gap from being
diluted by unrelated features.  The specs are still used to write the
human-readable reasons.

An Isolation Forest (no labels needed) learns what "normal" listings look like.
Listings that are easy to isolate get a high anomaly score, which is calibrated
to a 0-1 ``fraud_score``.  Listings priced at or above their fair value are
never labelled SUSPICIOUS (their score is damped) because the goal is to catch
suspiciously cheap deals.

Outputs
-------
    models/isolation_forest.pkl   (model + calibration + reference statistics)

Usage (from the project root, AFTER python src/train_model.py):
    python src/fraud_detection.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from predict import (
    FRIENDLY_NAMES,
    cached_load,
    predict_from_raw_row,
    prepare_raw_row,
    to_number,
)
from preprocess import (
    MODELS_DIR,
    TARGET_COLUMN,
    clean_data,
    fill_missing,
    load_artifacts,
    load_data,
    transform_features,
)

ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.pkl"

# Features the Isolation Forest looks at.
FRAUD_FEATURES = ["price_gap", TARGET_COLUMN]
# Extra columns whose training distribution is stored for the explanations.
REFERENCE_FEATURES = [
    "ram", "int_memory", "battery", "main_camera_mp", "selfie_camera_mp",
    "screen_size", "device_age", TARGET_COLUMN,
]

CONTAMINATION = 0.05          # expected share of anomalies in the training data
UNDERPRICE_TOLERANCE = 0.05   # price_gap must be below -0.05 to count as "under-priced"
DAMPING_FACTOR = 0.3          # score multiplier for listings NOT under-priced


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_fraud_frame(raw_df, price_gap, listed_price):
    """
    Assemble the feature frame (FRAUD_FEATURES order) the Isolation Forest expects.
    ``raw_df`` is accepted for API symmetry / future features; only the price gap
    and the listed price are used by the detector.
    """
    return pd.DataFrame({
        "price_gap": np.asarray(price_gap, dtype=float),
        TARGET_COLUMN: np.asarray(listed_price, dtype=float),
    })[FRAUD_FEATURES]


def calibrate_score(raw_score, lo, hi):
    """Map a raw anomaly score (higher = more anomalous) to the 0-1 range."""
    return float(np.clip((raw_score - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def percentile_of(sorted_values, value):
    """Percentile (0-100) of ``value`` inside an already sorted reference array."""
    return 100.0 * np.searchsorted(sorted_values, value, side="right") / len(sorted_values)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_fraud_detector():
    """
    Fit the Isolation Forest and save it (with calibration + reference stats)
    to models/isolation_forest.pkl.  Requires the preprocessing artifacts that
    train_model.py creates.
    """
    print("Loading data and preprocessing artifacts...")
    df = clean_data(load_data())
    scaler, encoders, meta = load_artifacts()
    X = transform_features(df, scaler, encoders, meta)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    # Fair price for each training listing, estimated out-of-fold so the model
    # never sees the row it is predicting (otherwise every gap would be ~0).
    print("Estimating fair prices (5-fold cross-validated Random Forest)...", end=" ", flush=True)
    forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    fair_price = cross_val_predict(forest, X, y, cv=5)
    print("Done.")

    filled = fill_missing(df, meta["fill_values"])
    price_gap = y - fair_price
    frame = build_fraud_frame(filled, price_gap, y)

    print("Training Isolation Forest...", end=" ", flush=True)
    iso = IsolationForest(
        n_estimators=300, contamination=CONTAMINATION, random_state=42, n_jobs=-1
    )
    iso.fit(frame)
    print("Done.")

    # Calibration: median listing -> 0, 99th-percentile anomaly -> 1.
    raw_scores = -iso.score_samples(frame)
    lo = float(np.median(raw_scores))
    hi = float(np.percentile(raw_scores, 99))
    threshold = calibrate_score(-iso.offset_, lo, hi)  # score above which iso says "outlier"

    bundle = {
        "model": iso,
        "features": FRAUD_FEATURES,
        "calib_lo": lo,
        "calib_hi": hi,
        "threshold": threshold,
        "reference": {c: np.sort(filled[c].to_numpy(dtype=float)) for c in REFERENCE_FEATURES},
        "gap_std": float(np.std(price_gap)),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ISOLATION_FOREST_PATH)

    scores = np.clip((raw_scores - lo) / (hi - lo + 1e-9), 0, 1)
    flagged = int(((scores >= threshold) & (price_gap < -UNDERPRICE_TOLERANCE)).sum())
    print(f"Saved Isolation Forest to {ISOLATION_FOREST_PATH}")
    print(f"  Fraud threshold (0-1 scale): {threshold:.3f}")
    print(f"  Training listings flagged SUSPICIOUS (under-priced anomalies): "
          f"{flagged} / {len(frame)} ({100 * flagged / len(frame):.1f}%)")
    return bundle


# --------------------------------------------------------------------------- #
# Scoring a single listing
# --------------------------------------------------------------------------- #
def load_fraud_bundle():
    """Load the saved Isolation Forest bundle (raises a helpful error if missing)."""
    return cached_load(ISOLATION_FOREST_PATH)


def explain_listing(raw_row, price_gap, expected_price, listed_price, bundle):
    """
    Build short human-readable reasons describing which features made the
    listing look normal or anomalous (price gap, high-end specs at a low price,
    out-of-range values ...).
    """
    ref = bundle["reference"]
    row = raw_row.iloc[0]
    reasons = []

    price_pct = percentile_of(ref[TARGET_COLUMN], listed_price)
    cheap = price_pct <= 25  # price is in the cheapest quarter of all listings

    # 1) Price versus the fair-price estimate.
    if price_gap < -UNDERPRICE_TOLERANCE:
        reasons.append(
            f"Listed price ({listed_price:.2f}) is {abs(price_gap):.2f} below the "
            f"fair price estimated from the specs ({expected_price:.2f})."
        )
    else:
        reasons.append(
            f"Listed price ({listed_price:.2f}) is in line with or above the "
            f"estimated fair price ({expected_price:.2f})."
        )

    # 2) High-end specs but a bottom-quartile price.
    if cheap:
        high_end = [
            FRIENDLY_NAMES[c] for c in ("ram", "int_memory", "battery", "main_camera_mp", "screen_size")
            if percentile_of(ref[c], row[c]) >= 75
        ]
        if high_end:
            reasons.append(
                "High-end specs (" + ", ".join(high_end) + ") but the price is in the "
                "cheapest quarter of all listings."
            )
        if row["five_g"] == "yes":
            reasons.append("5G-capable device priced in the cheapest quarter of listings.")
        if row["device_age"] <= 1:
            reasons.append("Very recent device priced like an old or low-end one.")

    # 3) Spec values outside the range seen in the data.
    for col in ("ram", "int_memory", "battery", "main_camera_mp", "selfie_camera_mp", "screen_size"):
        low, high = np.percentile(ref[col], [1, 99])
        if row[col] < low or row[col] > high:
            reasons.append(
                f"{FRIENDLY_NAMES[col].capitalize()} value is outside the typical range "
                "seen in the dataset."
            )

    return reasons


def score_listing(user_input, listed_price, fair_price_model="Random Forest"):
    """
    Score one listing.

    Parameters
    ----------
    user_input   : dict with the same keys as predict.predict_price
    listed_price : the asking price on the dataset's normalized scale
    Returns a dict: fraud_score (0-1), label (SUSPICIOUS / LEGITIMATE),
    reasons (list[str]), expected_price, price_gap, notes (list[str]).
    """
    price = to_number(listed_price)
    if price is None or price <= 0:
        raise ValueError("A valid (positive) listed price is required to check a listing.")

    bundle = load_fraud_bundle()
    raw_row, notes = prepare_raw_row(user_input)

    expected = predict_from_raw_row(raw_row, fair_price_model)
    gap = price - expected

    frame = build_fraud_frame(raw_row, [gap], [price])
    raw_score = float(-bundle["model"].score_samples(frame)[0])
    score = calibrate_score(raw_score, bundle["calib_lo"], bundle["calib_hi"])

    underpriced = gap < -UNDERPRICE_TOLERANCE
    if not underpriced:
        score *= DAMPING_FACTOR  # expensive / fairly priced listings are not scams of interest
    is_suspicious = underpriced and score >= bundle["threshold"]

    return {
        "fraud_score": float(np.clip(score, 0.0, 1.0)),
        "label": "SUSPICIOUS" if is_suspicious else "LEGITIMATE",
        "reasons": explain_listing(raw_row, gap, expected, price, bundle),
        "expected_price": expected,
        "price_gap": gap,
        "threshold": bundle["threshold"],
        "notes": notes,
    }


if __name__ == "__main__":
    train_fraud_detector()