"""
app.py
======
ResaleRadar - Streamlit front end.

Run from the project root with:
    streamlit run app/app.py

Tab 1  Price Estimator : predicts the fair (normalized) used price of a device.
Tab 2  Fraud Detector  : checks whether a listing's price looks suspiciously low.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the src/ package folder importable no matter where Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection import score_listing  # noqa: E402
from predict import MODEL_FILES, get_preprocessing_artifacts, predict_price  # noqa: E402
from preprocess import UNKNOWN_TOKEN  # noqa: E402

st.set_page_config(page_title="ResaleRadar", page_icon="📡", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_options():
    """Read the available brands / OS values and price statistics from the saved artifacts."""
    _, encoders, meta = get_preprocessing_artifacts()
    brands = sorted(c for c in encoders["device_brand"].classes_ if c != UNKNOWN_TOKEN)
    systems = sorted(c for c in encoders["os"].classes_ if c != UNKNOWN_TOKEN)
    return brands, systems, meta


def render_input_fields(prefix, brands, systems):
    """
    Draw the device input widgets (shared by both tabs) and return the values as a dict.
    ``prefix`` keeps the widget keys of the two tabs separate.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        default_brand = brands.index("Samsung") if "Samsung" in brands else 0
        brand = st.selectbox("Device brand", brands, index=default_brand, key=f"{prefix}_brand")
        default_os = systems.index("Android") if "Android" in systems else 0
        os_name = st.selectbox("Operating system", systems, index=default_os, key=f"{prefix}_os")
        screen = st.number_input("Screen size (inches)", min_value=1.0, max_value=20.0,
                                 value=6.1, step=0.1, key=f"{prefix}_screen")
        age = st.number_input("Device age (years since launch)", min_value=0.0, max_value=15.0,
                              value=2.0, step=0.5, key=f"{prefix}_age")

    with col2:
        four_g = st.radio("4G", ["Yes", "No"], horizontal=True, key=f"{prefix}_4g")
        five_g = st.radio("5G", ["No", "Yes"], horizontal=True, key=f"{prefix}_5g")
        front = st.number_input("Front camera (MP)", min_value=0.0, max_value=100.0,
                                value=8.0, step=1.0, key=f"{prefix}_front")
        rear = st.number_input("Rear camera (MP)", min_value=0.0, max_value=200.0,
                               value=13.0, step=1.0, key=f"{prefix}_rear")

    with col3:
        ram = st.number_input("RAM (GB)", min_value=0.0, max_value=32.0,
                              value=4.0, step=1.0, key=f"{prefix}_ram")
        battery = st.number_input("Battery capacity (mAh)", min_value=0.0, max_value=15000.0,
                                  value=4000.0, step=100.0, key=f"{prefix}_battery")
        storage = st.number_input("Storage (GB)", min_value=0.0, max_value=2048.0,
                                  value=64.0, step=16.0, key=f"{prefix}_storage")

    # Optional extra features of the dataset; 0 means "use the dataset median".
    with st.expander("Optional details (leave at 0 to use dataset medians)"):
        o1, o2, o3 = st.columns(3)
        weight = o1.number_input("Weight (g)", min_value=0.0, max_value=2000.0,
                                 value=0.0, step=10.0, key=f"{prefix}_weight")
        days_used = o2.number_input("Days used", min_value=0.0, max_value=5000.0,
                                    value=0.0, step=10.0, key=f"{prefix}_days")
        new_price = o3.number_input("New price when launched (normalized)", min_value=0.0,
                                    max_value=10.0, value=0.0, step=0.1, key=f"{prefix}_newprice")

    return {
        "device_brand": brand,
        "os": os_name,
        "screen_size_inches": screen,
        "four_g": four_g.lower(),
        "five_g": five_g.lower(),
        "selfie_camera_mp": front,
        "main_camera_mp": rear,
        "ram": ram,
        "battery": battery,
        "int_memory": storage,
        "device_age": age,
        # 0 -> None so the backend falls back to the dataset median.
        "weight": weight or None,
        "days_used": days_used or None,
        "normalized_new_price": new_price or None,
    }


def show_notes(notes):
    """Display any warnings returned by the backend (defaults used, unseen categories ...)."""
    for note in notes:
        st.info(note)


# --------------------------------------------------------------------------- #
# Page header + artifact check
# --------------------------------------------------------------------------- #
st.title("📡 ResaleRadar")
st.caption("Fair-price estimation and fraud detection for second-hand electronics")

try:
    BRANDS, SYSTEMS, META = load_options()
except FileNotFoundError:
    st.error(
        "Trained models were not found. Please run these commands from the project "
        "root first:\n\n`python src/train_model.py`\n\n`python src/fraud_detection.py`"
    )
    st.stop()

tab_price, tab_fraud = st.tabs(["💰 Price Estimator", "🕵️ Fraud Detector"])

# --------------------------------------------------------------------------- #
# Tab 1: Price Estimator
# --------------------------------------------------------------------------- #
with tab_price:
    st.subheader("Estimate the fair price of a used device")
    inputs_price = render_input_fields("price", BRANDS, SYSTEMS)
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()), index=1, key="model_choice")

    if st.button("Estimate Price", type="primary", key="btn_estimate"):
        try:
            result = predict_price(inputs_price, model_name)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # any unexpected failure is shown instead of crashing
            st.error(f"Could not estimate the price: {exc}")
        else:
            show_notes(result["notes"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted normalized price", f"{result['predicted_price']:.3f}")
            c2.metric("Price band (±10%)", f"{result['lower']:.3f} – {result['upper']:.3f}")
            c3.metric("Confidence label", result["segment"])
            low, high = result["price_tiers"]
            st.caption(
                f"Model: {result['model']}. Budget < {low:.2f} ≤ Mid-range < {high:.2f} ≤ Premium "
                "(tiers are the price terciles of the training data)."
            )

    comparison_path = PROJECT_ROOT / "models" / "model_comparison.csv"
    if comparison_path.exists():
        with st.expander("Model performance (test set)"):
            st.dataframe(pd.read_csv(comparison_path).round(4), hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 2: Fraud Detector
# --------------------------------------------------------------------------- #
with tab_fraud:
    st.subheader("Check whether a listing looks suspicious")
    inputs_fraud = render_input_fields("fraud", BRANDS, SYSTEMS)

    # The listed price is required: the detector looks for prices that are too low for the specs.
    listed_price = st.number_input(
        "Listed price (normalized, same scale as the dataset)",
        min_value=0.0, max_value=15.0, value=float(round(META["target_median"], 2)), step=0.05,
        key="fraud_price",
        help=f"In the training data the normalized used price ranges from "
             f"{META['target_min']:.2f} to {META['target_max']:.2f} "
             f"(median {META['target_median']:.2f}).",
    )

    if st.button("Check Listing", type="primary", key="btn_check"):
        try:
            verdict = score_listing(inputs_fraud, listed_price)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Could not check the listing: {exc}")
        else:
            show_notes(verdict["notes"])
            score = float(min(max(verdict["fraud_score"], 0.0), 1.0))
            st.progress(score, text=f"Fraud score: {score:.2f} (0 = safe, 1 = very suspicious)")

            if verdict["label"] == "SUSPICIOUS":
                st.error("🚨 SUSPICIOUS")
            else:
                st.success("✅ LEGITIMATE")

            st.markdown("**Why?**")
            for reason in verdict["reasons"]:
                st.markdown(f"- {reason}")
            st.caption(
                f"A listing is flagged SUSPICIOUS when it is under-priced and its score is at least "
                f"{verdict['threshold']:.2f}."
            )