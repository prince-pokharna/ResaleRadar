# 📡 ResaleRadar

**An Intelligent Fair-Price Estimator and Fraud Detection System for Second-Hand Electronics.**

---

## Problem Statement

Buying a used phone or tablet is risky. Buyers struggle to know whether an asking price is fair for the
specifications on offer, and sellers (or scammers) can post misleading listings, especially
"too good to be true" prices designed to lure victims.

ResaleRadar tackles both problems with machine learning:

1. **Price estimation** - three regression models (Linear Regression, Random Forest, XGBoost) predict the
   fair *normalized used price* of a device from its specifications, with a ±10 % price band and a
   Budget / Mid-range / Premium label.
2. **Fraud detection** - an unsupervised **Isolation Forest** flags listings whose price is suspiciously low
   relative to their specs, returning a fraud score (0-1), a `SUSPICIOUS` / `LEGITIMATE` label and a short
   explanation.

Everything runs fully offline once the dataset has been downloaded.

## Dataset

**Used Phones & Tablets Pricing Dataset** (Used Handheld Device Data) by *ahsan81* on Kaggle:
<https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data>

Download `used_device_data.csv` and place it in the `data/` folder.

| Column | Meaning |
|---|---|
| `device_brand`, `os` | Brand and operating system |
| `screen_size` | Screen size in **cm** (the app takes inches and converts) |
| `4g`, `5g` | 4G / 5G availability (yes / no) |
| `main_camera_mp`, `selfie_camera_mp` | Rear / front camera in MP |
| `int_memory`, `ram` | Storage and RAM in GB |
| `battery`, `weight` | Battery (mAh) and weight (g) |
| `release_year`, `days_used` | Launch year and days the device was used |
| `normalized_new_price` | Normalized price when new |
| `normalized_used_price` | **Target** - normalized used price |

`release_year` is converted into the engineered feature `device_age` (= 2021 − release year).

## Tech Stack

- **Python 3.10+**
- **pandas, NumPy** - data handling
- **scikit-learn** - preprocessing, Linear Regression, Random Forest, Isolation Forest
- **XGBoost** - gradient boosting regressor
- **matplotlib, seaborn** - EDA visualisations
- **Streamlit** - web front end
- **joblib** - model persistence (`.pkl`)

## Folder Structure

```
ResaleRadar/
├── data/
│   └── used_device_data.csv          ← you place this manually
├── notebooks/
│   └── EDA.ipynb                     ← Exploratory Data Analysis notebook
├── src/
│   ├── preprocess.py                 ← Data cleaning + feature engineering
│   ├── train_model.py                ← Trains all 3 models
│   ├── fraud_detection.py            ← Isolation Forest fraud scorer
│   └── predict.py                    ← Single prediction utility
├── models/                           ← auto-created when training runs
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── preprocess_meta.pkl           ← fill values, price tiers (used by the app)
│   └── model_comparison.csv
├── app/
│   └── app.py                        ← Streamlit frontend
├── requirements.txt
└── README.md
```

## How to Install and Run

1. **Clone / unzip** the project and open a terminal in the `ResaleRadar/` folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**: download `used_device_data.csv` from Kaggle and put it in `data/`.

5. **Train the price models** (creates the scaler, encoders and the three `.pkl` models plus
   `models/model_comparison.csv`; takes well under 5 minutes):
   ```bash
   python src/train_model.py
   ```

6. **Train the fraud detector** (must run *after* step 5):
   ```bash
   python src/fraud_detection.py
   ```

7. **Launch the app**:
   ```bash
   streamlit run app/app.py
   ```

8. *(Optional)* **Explore the data**: `jupyter notebook notebooks/EDA.ipynb`
   (run step 5 first so the feature-importance cell can load the trained Random Forest).

*(Optional)* Quick command-line prediction demo: `python src/predict.py`.

## How It Works

**Preprocessing** - missing values are filled with the median (numeric) or mode (categorical);
categorical columns are label-encoded (with an extra "unknown" class so unseen brands never crash the app);
numeric columns are standardised. The scaler and encoders are fitted on the training split only.

**Price estimation** - the selected model predicts the normalized used price. The app shows the prediction,
a ±10 % band and a tier label (Budget / Mid-range / Premium, defined by the price terciles of the training data).

**Fraud detection** - a Random Forest estimates the *fair* price of every listing (out-of-fold, so the
estimate is honest). The Isolation Forest is then trained on `price_gap = listed − fair` and the listed
price. Listings that are easy to isolate receive a high anomaly score, which is calibrated to 0-1.
A listing is `SUSPICIOUS` only when it is **under-priced** and the score exceeds the Isolation Forest's own
outlier threshold; reasons (price gap, high-end specs at a low price, out-of-range values) are shown alongside.

## Notes and Design Decisions

- **Screen size**: the dataset stores centimetres; the app asks for inches (as required) and converts ×2.54.
- **Fraud tab needs a price**: a "suspiciously low price" check needs the listed price, so the Fraud tab has
  one extra input, *Listed price*, on the dataset's normalized scale (the helper text shows the typical range).
- **Optional details**: `weight`, `days_used` and `normalized_new_price` are also training features but are not
  in the required form, so they sit in an optional expander. When left at 0 the dataset median is used.
  Providing the launch price noticeably improves accuracy.
- **Edge cases**: missing/invalid inputs fall back to medians/modes, unseen brands or OS values are treated as
  "unknown", and the app tells the user whenever a default was used.

## Screenshots

> _Add screenshots here after running the app._

| Price Estimator | Fraud Detector |
|---|---|
| `![Price Estimator](screenshots/price_estimator.png)` | `![Fraud Detector](screenshots/fraud_detector.png)` |

## Credits

Minor project submitted in partial fulfilment of the requirements of the degree program.

**Project:** ResaleRadar  
**Student:** _Your Name_ (Roll No. _XXXXXXX_)  
**Guide:** _Guide Name_  
**Department / University:** _Department of Computer Science, Your University_  
**Academic Year:** 2026-27