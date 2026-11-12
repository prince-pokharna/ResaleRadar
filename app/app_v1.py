"""ResaleRadar - Fair-price estimator and fraud detector for used phones/tablets."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

st.set_page_config(page_title="ResaleRadar", page_icon="📡", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "used_device_data.csv"
TARGET = "normalized_used_price"
DROP_COLS = ["normalized_new_price"]  # a real seller would not know this value

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem;}
.hero {
    padding: 28px 32px; border-radius: 20px; margin-bottom: 18px;
    background: linear-gradient(120deg, #7C5CFF 0%, #00C2FF 100%);
    box-shadow: 0 10px 30px rgba(124, 92, 255, .35);
}
.hero h1 {margin: 0; color: white; font-size: 2.4rem;}
.hero p {margin: 6px 0 0 0; color: rgba(255,255,255,.92); font-size: 1.05rem;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    padding: 14px 18px; border-radius: 16px;
}
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    padding: 18px 22px; border-radius: 16px; margin-bottom: 12px;
}
.badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-weight: 700; font-size: 1.05rem; color: white;
}
.stTabs [data-baseweb="tab"] {font-size: 1.05rem; padding: 10px 18px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>📡 ResaleRadar</h1>
  <p>AI-powered fair-price estimation and fraud detection for second-hand phones &amp; tablets</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- data
if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nPlace used_device_data.csv inside the data/ folder.")
    st.stop()


@st.cache_data
def load_data():
    """Load the CSV and fill missing values (median for numbers, mode for text)."""
    data = pd.read_csv(DATA_PATH)
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])
    return data


df = load_data()
if TARGET not in df.columns:
    st.error(f"Column '{TARGET}' not found. Your columns are: {list(df.columns)}")
    st.stop()

FEATURES = [c for c in df.columns if c != TARGET and c not in DROP_COLS]
CATS = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
NUMS = [c for c in FEATURES if c not in CATS]


def pretty(name):
    """Turn column names like 'back_camera_mp' into 'Back Camera Mp'."""
    return name.replace("_", " ").title()


def encode(frame, columns=None):
    """One-hot encode categories; align to the training columns when given."""
    X = pd.get_dummies(frame[FEATURES], columns=CATS).astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


@st.cache_resource(show_spinner="Training models for the first time (about 10-20 seconds)...")
def build_models():
    """Train 3 regressors + Isolation Forest once and cache them."""
    X = encode(df)
    y = df[TARGET]
    cols = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    }
    rows, test_preds = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        test_preds[name] = pred
        rows.append({
            "Model": name,
            "MAE": round(mean_absolute_error(y_te, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
            "R2": round(r2_score(y_te, pred), 4),
        })
    metrics = pd.DataFrame(rows)

    X_iso = X.copy()
    X_iso["__price"] = y.values
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(X_iso)
    train_scores = iso.score_samples(X_iso)
    return models, metrics, cols, iso, train_scores, y_te, test_preds


models, metrics, cols, iso, train_scores, y_te, test_preds = build_models()

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## ⚙️ Settings")
model_name = st.sidebar.selectbox("Prediction model", list(models.keys()), index=2)
currency = st.sidebar.radio("Display currency", ["EUR (€)", "INR (₹)"], horizontal=True)
if currency.startswith("EUR"):
    rate, sym = 1.0, "€"
else:
    rate = st.sidebar.number_input("EUR → INR rate", 50.0, 200.0, 90.0)
    sym = "₹"

st.sidebar.markdown("## 📱 Device Specs")
spec = {}
for c in CATS:
    spec[c] = st.sidebar.selectbox(pretty(c), sorted(df[c].unique()))
for c in NUMS:
    lo, hi, med = df[c].min(), df[c].max(), df[c].median()
    if pd.api.types.is_integer_dtype(df[c]):
        spec[c] = st.sidebar.slider(pretty(c), int(lo), int(hi), int(med))
    else:
        step = max(round(float((hi - lo) / 200), 2), 0.01)
        spec[c] = st.sidebar.slider(pretty(c), float(lo), float(hi), float(med), step=step)


def money(norm_value):
    """Convert the normalized (log-scale) value into a displayed price."""
    return float(np.exp(norm_value)) * rate


# ---------------------------------------------------------------- predictions
row = pd.DataFrame([spec])
X_row = encode(row, cols)
all_preds = {name: float(m.predict(X_row)[0]) for name, m in models.items()}
fair_norm = all_preds[model_name]
fair_price = money(fair_norm)
band_lo, band_hi = fair_price * 0.9, fair_price * 1.1
percentile = float((df[TARGET] < fair_norm).mean() * 100)
if percentile < 33:
    tier, tier_color = "Budget", "#22C55E"
elif percentile < 66:
    tier, tier_color = "Mid-range", "#F59E0B"
else:
    tier, tier_color = "Premium", "#EF4444"

# ---------------------------------------------------------------- KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings analysed", f"{len(df):,}")
k2.metric("Brands covered", df["device_brand"].nunique() if "device_brand" in df.columns else "-")
k3.metric("Best model R²", f"{metrics['R2'].max():.3f}")
k4.metric("Active model", model_name)

tab1, tab2, tab3, tab4 = st.tabs(
    ["💰 Price Estimator", "🚨 Fraud Detector", "📊 Market Explorer", "🧠 Model Performance"]
)

# ---------------------------------------------------------------- TAB 1
with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fair_price,
            number={"prefix": sym, "valueformat": ",.0f", "font": {"size": 48}},
            title={"text": "Estimated Fair Price"},
            gauge={
                "axis": {"range": [money(df[TARGET].min()), money(df[TARGET].max())]},
                "bar": {"color": "#7C5CFF"},
                "steps": [{"range": [band_lo, band_hi], "color": "rgba(0,194,255,.35)"}],
            },
        ))
        gauge.update_layout(template="plotly_dark", height=340, margin=dict(t=60, b=10, l=30, r=30),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        st.markdown(
            f"""
<div class="card">
  <h4 style="margin-top:0">Fair price band (±10%)</h4>
  <h2 style="margin:0">{sym}{band_lo:,.0f} – {sym}{band_hi:,.0f}</h2>
</div>
<div class="card">
  <h4 style="margin-top:0">Market tier</h4>
  <span class="badge" style="background:{tier_color}">{tier}</span>
  <p style="margin-top:12px">Priced higher than <b>{percentile:.0f}%</b> of devices in the dataset.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🤝 What each model predicts")
    cmp_df = pd.DataFrame({"Model": list(all_preds), "Price": [money(v) for v in all_preds.values()]})
    fig = px.bar(cmp_df, x="Model", y="Price", color="Model", text_auto=",.0f", template="plotly_dark",
                 color_discrete_sequence=["#7C5CFF", "#00C2FF", "#22C55E"])
    fig.update_layout(showlegend=False, height=330, paper_bgcolor="rgba(0,0,0,0)",
                      yaxis_title=f"Predicted price ({sym})")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.markdown("#### Enter the seller's asking price to check the listing")
    asking = st.number_input(f"Asking price ({sym})", min_value=1.0,
                             value=float(round(fair_price)), step=float(max(round(fair_price * 0.05), 1)))

    asking_norm = float(np.log(asking / rate))
    X_iso_row = X_row.copy()
    X_iso_row["__price"] = asking_norm
    iso_score = float(iso.score_samples(X_iso_row)[0])
    iso_component = float((train_scores > iso_score).mean())  # high = rarer than most listings

    ratio = (asking / rate) / float(np.exp(fair_norm))
    gap = 1 - ratio
    gap_component = float(np.clip(gap / 0.6, 0, 1)) if gap > 0 else 0.0
    fraud_score = 0.6 * gap_component + 0.4 * iso_component

    if ratio > 1.3:
        verdict, vcolor = "💸 OVERPRICED", "#F59E0B"
    elif fraud_score >= 0.6:
        verdict, vcolor = "🚨 SUSPICIOUS", "#EF4444"
    elif fraud_score >= 0.35:
        verdict, vcolor = "⚠️ CAUTION", "#F59E0B"
    else:
        verdict, vcolor = "✅ LEGITIMATE", "#22C55E"

    c1, c2 = st.columns([1, 1])
    with c1:
        g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_score * 100,
            number={"suffix": " / 100", "valueformat": ".0f"},
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7C5CFF"},
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,197,94,.35)"},
                    {"range": [35, 60], "color": "rgba(245,158,11,.35)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,.35)"},
                ],
            },
        ))
        g2.update_layout(template="plotly_dark", height=320, margin=dict(t=60, b=10, l=30, r=30),
                         paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g2, use_container_width=True)
    with c2:
        st.markdown(
            f"""
<div class="card">
  <span class="badge" style="background:{vcolor}">{verdict}</span>
  <p style="margin-top:14px">Fair price: <b>{sym}{fair_price:,.0f}</b><br>
  Asking price: <b>{sym}{asking:,.0f}</b> ({ratio * 100:.0f}% of fair price)</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Why this result?**")
        reasons = []
        if gap > 0.4:
            reasons.append(f"Asking price is {gap * 100:.0f}% below the estimated fair price, which is a classic scam signal.")
        elif gap > 0.2:
            reasons.append(f"Asking price is {gap * 100:.0f}% below fair price, so it is a bargain or a bait.")
        if ratio > 1.3:
            reasons.append(f"Asking price is {abs(gap) * 100:.0f}% above the fair price, so the seller may be overcharging.")
        if iso_component > 0.9:
            reasons.append("This combination of specs and price is very rare in real market data (Isolation Forest).")
        if not reasons:
            reasons.append("Price and specs are consistent with typical listings in the market data.")
        for r in reasons:
            st.write("• " + r)

# ---------------------------------------------------------------- TAB 3
with tab3:
    template = "plotly_dark"
    a, b = st.columns(2)
    with a:
        if "device_brand" in df.columns:
            top = df["device_brand"].value_counts().head(12).index
            fig = px.box(df[df["device_brand"].isin(top)], x="device_brand", y=TARGET, color="device_brand",
                         template=template, title="Price distribution by brand (top 12)")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with b:
        if "ram" in df.columns:
            color_col = "os" if "os" in df.columns else None
            fig = px.scatter(df, x="ram", y=TARGET, color=color_col, opacity=0.6, template=template,
                             title="RAM vs price")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    with c:
        if "days_used" in df.columns:
            fig = px.scatter(df, x="days_used", y=TARGET, opacity=0.5, template=template,
                             title="Depreciation: days used vs price")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with d:
        corr = df[NUMS + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis", template=template,
                        title="Correlation heatmap")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.markdown("#### Model comparison (20% hold-out test set)")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    p1, p2 = st.columns(2)
    with p1:
        fig = px.bar(metrics, x="Model", y="R2", color="Model", text_auto=".3f", template="plotly_dark",
                     title="R² score (higher is better)",
                     color_discrete_sequence=["#7C5CFF", "#00C2FF", "#22C55E"])
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        imp = pd.Series(models["Random Forest"].feature_importances_, index=cols).nlargest(10).sort_values()
        fig = px.bar(imp, orientation="h", template="plotly_dark", title="Top 10 features (Random Forest)")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    lo_v, hi_v = float(y_te.min()), float(y_te.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_te, y=test_preds[model_name], mode="markers", name="Predictions",
                             marker=dict(color="#00C2FF", opacity=0.6)))
    fig.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="Perfect fit",
                             line=dict(color="#EF4444", dash="dash")))
    fig.update_layout(template="plotly_dark", title=f"Actual vs predicted: {model_name}",
                      xaxis_title="Actual (normalized)", yaxis_title="Predicted (normalized)",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("ResaleRadar · University Minor Project · Dataset: Used Handheld Device Data (Kaggle, ahsan81)")