"""ResaleRadar v2 - Fair-price estimator, fraud detector and bulk checker for used phones/tablets."""
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

# Friendly names and short hints for sidebar inputs (unknown columns fall back automatically)
LABELS = {
    "device_brand": "Brand", "os": "Operating system", "screen_size": "Screen size (cm)",
    "4g": "4G support", "5g": "5G support", "main_camera_mp": "Rear camera (MP)",
    "selfie_camera_mp": "Front camera (MP)", "int_memory": "Storage (GB)", "ram": "RAM (GB)",
    "battery": "Battery (mAh)", "weight": "Weight (g)", "release_year": "Release year",
    "days_used": "Days used",
}
HELP = {
    "days_used": "How many days the device has already been used.",
    "screen_size": "Diagonal screen size in centimetres.",
    "release_year": "Year the model was launched.",
}

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
.tip {
    border-left: 4px solid #7C5CFF; background: rgba(124,92,255,.10);
    padding: 10px 16px; border-radius: 8px; margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; padding: 10px 16px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>📡 ResaleRadar</h1>
  <p>Know the fair price of a used phone or tablet, and spot suspicious listings before you pay</p>
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


def label(name):
    """Friendly label for a column name."""
    return LABELS.get(name, name.replace("_", " ").title())


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
    train_scores = np.sort(iso.score_samples(X_iso))
    return models, metrics, cols, iso, train_scores, y_te, test_preds


models, metrics, cols, iso, train_scores, y_te, test_preds = build_models()


def fraud_scores(X, asking_norm, fair_norm):
    """Fraud risk (0-1) = 60% price gap below fair price + 40% Isolation Forest rarity."""
    Xi = X.copy()
    Xi["__price"] = asking_norm
    s = iso.score_samples(Xi)
    iso_comp = 1 - np.searchsorted(train_scores, s, side="right") / len(train_scores)
    ratio = np.atleast_1d(np.exp(np.asarray(asking_norm) - np.asarray(fair_norm)))
    gap = 1 - ratio
    gap_comp = np.where(gap > 0, np.clip(gap / 0.6, 0, 1), 0.0)
    score = 0.6 * gap_comp + 0.4 * iso_comp
    return np.atleast_1d(score), ratio, np.atleast_1d(iso_comp)


def verdict_for(ratio, score):
    """Turn price ratio and fraud score into a label and colour."""
    if ratio > 1.3:
        return "💸 OVERPRICED", "#F59E0B"
    if score >= 0.6:
        return "🚨 SUSPICIOUS", "#EF4444"
    if score >= 0.35:
        return "⚠️ CAUTION", "#F59E0B"
    return "✅ LEGITIMATE", "#22C55E"


NEXT_STEPS = {
    "🚨 SUSPICIOUS": "Do NOT pay in advance. Meet in person, check the IMEI number, and ask for the original bill and box.",
    "⚠️ CAUTION": "Ask for the bill and more real photos, and test the device fully before paying.",
    "✅ LEGITIMATE": "The price looks normal. Still inspect the device in person before paying.",
    "💸 OVERPRICED": "Negotiate. Use the fair price range from the Fair Price tab as your argument.",
}

# ---------------------------------------------------------------- sidebar state
def default_value(c):
    """Starting value for a sidebar input (most common option / median number)."""
    if c in CATS:
        return df[c].mode()[0]
    if pd.api.types.is_integer_dtype(df[c]):
        return int(df[c].median())
    return float(df[c].median())


for c in FEATURES:
    if f"w_{c}" not in st.session_state:
        st.session_state[f"w_{c}"] = default_value(c)


def reset_specs():
    """Put every sidebar input back to its default."""
    for c in FEATURES:
        st.session_state[f"w_{c}"] = default_value(c)


def random_listing():
    """Fill the sidebar with a real listing from the dataset."""
    r = df.sample(1).iloc[0]
    for c in FEATURES:
        if c in CATS:
            st.session_state[f"w_{c}"] = str(r[c])
        elif pd.api.types.is_integer_dtype(df[c]):
            st.session_state[f"w_{c}"] = int(r[c])
        else:
            st.session_state[f"w_{c}"] = float(r[c])


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## 👋 New here?")
st.sidebar.caption("Open the **Start Here** tab. Or click the button below to try a real example.")
st.sidebar.button("🎲 Try a real example", on_click=random_listing, use_container_width=True)
st.sidebar.button("↩️ Reset specs", on_click=reset_specs, use_container_width=True)

st.sidebar.markdown("## ⚙️ Settings")
model_name = st.sidebar.selectbox("Prediction model", list(models.keys()), index=2,
                                  help="XGBoost is usually the most accurate. See the Model Performance tab.")
currency = st.sidebar.radio("Display currency", ["EUR (€)", "INR (₹)"], horizontal=True)
if currency.startswith("EUR"):
    rate, sym = 1.0, "€"
else:
    rate = st.sidebar.number_input("EUR → INR rate", 50.0, 200.0, 90.0)
    sym = "₹"

st.sidebar.markdown("## 📱 Device Specs")
spec = {}
for c in CATS:
    spec[c] = st.sidebar.selectbox(label(c), sorted(df[c].unique()), key=f"w_{c}", help=HELP.get(c))
for c in NUMS:
    lo, hi = df[c].min(), df[c].max()
    if lo == hi:
        spec[c] = lo
        continue
    if pd.api.types.is_integer_dtype(df[c]):
        spec[c] = st.sidebar.slider(label(c), int(lo), int(hi), step=1, key=f"w_{c}", help=HELP.get(c))
    else:
        step = max(round(float((hi - lo) / 200), 2), 0.01)
        spec[c] = st.sidebar.slider(label(c), float(lo), float(hi), step=step, key=f"w_{c}", help=HELP.get(c))


def money(norm_value):
    """Convert the normalized (log-scale) value into a displayed price."""
    return float(np.exp(norm_value)) * rate


# ---------------------------------------------------------------- predictions
row = pd.DataFrame([spec])
X_row = encode(row, cols)
all_preds = {name: float(m.predict(X_row)[0]) for name, m in models.items()}
fair_norm = all_preds[model_name]
fair_price = money(fair_norm)

# Price range from how much the Random Forest trees disagree (smart confidence band)
rf = models["Random Forest"]
tree_preds = np.array([t.predict(X_row.values)[0] for t in rf.estimators_])
half = max(float(np.percentile(tree_preds, 90) - np.percentile(tree_preds, 10)) / 2, 0.05)
band_lo, band_hi = money(fair_norm - half), money(fair_norm + half)
confidence = "High" if half < 0.08 else "Medium" if half < 0.15 else "Low"

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

t_help, t_price, t_fraud, t_bulk, t_market, t_perf = st.tabs(
    ["🏠 Start Here", "💰 Fair Price", "🚨 Fraud Check", "📦 Bulk Check", "📊 Market Explorer", "🧠 Model Performance"]
)

# ---------------------------------------------------------------- START HERE
with t_help:
    st.markdown("### Welcome! ResaleRadar answers two questions")
    q1, q2 = st.columns(2)
    q1.markdown('<div class="card"><h4 style="margin-top:0">💰 What should it cost?</h4>'
                'Get the fair price of a used phone or tablet.</div>', unsafe_allow_html=True)
    q2.markdown('<div class="card"><h4 style="margin-top:0">🚨 Is this listing safe?</h4>'
                'Type the seller\'s price and see if it looks like a scam.</div>', unsafe_allow_html=True)

    st.markdown("### How to use it in 3 steps")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="card"><b>1. Enter the device</b><br>Use the left sidebar, or click '
                '<b>🎲 Try a real example</b>.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="card"><b>2. Check the price</b><br>Open <b>💰 Fair Price</b> to see what it '
                'should cost and how to negotiate.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="card"><b>3. Check the seller</b><br>Open <b>🚨 Fraud Check</b>, type the asking '
                'price and read the verdict.</div>', unsafe_allow_html=True)

    st.markdown("### Quick glossary")
    st.markdown(
        "- **Fair price**: what similar used devices normally sell for.\n"
        "- **Fraud score**: 0–100. Under 35 is safe, 35–60 needs caution, above 60 is suspicious.\n"
        "- **Tier**: Budget, Mid-range or Premium, compared with all devices in the data.\n"
        "- **Bulk Check**: upload a CSV with many listings and check them all at once."
    )
    st.info("Prices are estimates learned from a public dataset, shown in euros by default. "
            "Use them as a guide, not a guarantee.")

# ---------------------------------------------------------------- FAIR PRICE
with t_price:
    st.markdown('<div class="tip">💡 Set the device specs in the sidebar. The fair price updates instantly.</div>',
                unsafe_allow_html=True)
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
  <h4 style="margin-top:0">Likely price range</h4>
  <h2 style="margin:0">{sym}{band_lo:,.0f} – {sym}{band_hi:,.0f}</h2>
  <p style="margin:8px 0 0 0">Model confidence: <b>{confidence}</b></p>
</div>
<div class="card">
  <h4 style="margin-top:0">Market tier</h4>
  <span class="badge" style="background:{tier_color}">{tier}</span>
  <p style="margin-top:12px">Priced higher than <b>{percentile:.0f}%</b> of devices in the data.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🤝 Smart advice")
    role = st.radio("I am a...", ["Buyer", "Seller"], horizontal=True, key="role")
    if role == "Buyer":
        st.success(f"Open your offer near **{sym}{fair_price * 0.90:,.0f}**. "
                   f"Do not pay more than **{sym}{band_hi:,.0f}**.")
    else:
        st.success(f"List it at about **{sym}{fair_price * 1.08:,.0f}** to leave room to negotiate. "
                   f"Do not accept less than **{sym}{band_lo:,.0f}**.")

    # Depreciation simulator
    if "days_used" in NUMS:
        st.markdown("#### 📉 How the price drops with use")
        dmax = int(df["days_used"].max())
        days = np.linspace(0, dmax, 30).astype(int)
        sim = pd.concat([row] * len(days), ignore_index=True)
        sim["days_used"] = days
        sim_price = [money(v) for v in models[model_name].predict(encode(sim, cols))]
        fig = px.line(x=days, y=sim_price, template="plotly_dark",
                      labels={"x": "Days used", "y": f"Estimated price ({sym})"})
        fig.update_traces(line_color="#00C2FF")
        fig.add_vline(x=int(spec["days_used"]), line_dash="dash", line_color="#7C5CFF")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)

        later = row.copy()
        later["days_used"] = min(int(spec["days_used"]) + 180, dmax)
        later_price = money(models[model_name].predict(encode(later, cols))[0])
        change = (later_price / fair_price - 1) * 100 if fair_price else 0
        st.metric("Expected price in 6 months", f"{sym}{later_price:,.0f}", f"{change:+.1f}%")

    # Similar real devices
    st.markdown("#### 🔎 Similar real devices in the data")
    if NUMS:
        std = df[NUMS].std().replace(0, 1)
        z = (df[NUMS] - df[NUMS].mean()) / std
        z_row = (row[NUMS].astype(float) - df[NUMS].mean()) / std
        dist = np.sqrt(((z - z_row.iloc[0]) ** 2).sum(axis=1))
        near = df.loc[dist.nsmallest(5).index]
        show_cols = [c for c in ["device_brand", "os", "ram", "int_memory", "battery", "release_year", "days_used"]
                     if c in near.columns]
        table = near[show_cols].copy()
        table.columns = [label(c) for c in show_cols]
        table[f"Real price ({sym})"] = [round(money(v)) for v in near[TARGET]]
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### 🧪 What each model predicts")
    cmp_df = pd.DataFrame({"Model": list(all_preds), "Price": [money(v) for v in all_preds.values()]})
    fig = px.bar(cmp_df, x="Model", y="Price", color="Model", text_auto=",.0f", template="plotly_dark",
                 color_discrete_sequence=["#7C5CFF", "#00C2FF", "#22C55E"])
    fig.update_layout(showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)",
                      yaxis_title=f"Predicted price ({sym})")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- FRAUD CHECK
with t_fraud:
    st.markdown('<div class="tip">💡 Set the device specs in the sidebar, then type the price the seller is asking.</div>',
                unsafe_allow_html=True)
    asking = st.number_input(f"Seller's asking price ({sym})", min_value=1.0,
                             value=float(max(round(fair_price), 1)),
                             step=float(max(round(fair_price * 0.05), 1)))

    asking_norm = float(np.log(asking / rate))
    sc, ra, io = fraud_scores(X_row, asking_norm, fair_norm)
    fraud_score, ratio, iso_component = float(sc[0]), float(ra[0]), float(io[0])
    gap = 1 - ratio
    verdict, vcolor = verdict_for(ratio, fraud_score)

    c1, c2 = st.columns(2)
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
            reasons.append(f"Asking price is {gap * 100:.0f}% below the fair price, a classic scam signal.")
        elif gap > 0.2:
            reasons.append(f"Asking price is {gap * 100:.0f}% below fair price: a bargain or a bait.")
        if ratio > 1.3:
            reasons.append(f"Asking price is {abs(gap) * 100:.0f}% above fair price, so the seller may be overcharging.")
        if iso_component > 0.9:
            reasons.append("This mix of specs and price is very rare in real market data.")
        if not reasons:
            reasons.append("Price and specs are consistent with typical listings.")
        for r in reasons:
            st.write("• " + r)
        st.markdown("**What to do next**")
        st.info(NEXT_STEPS[verdict])

# ---------------------------------------------------------------- BULK CHECK
with t_bulk:
    st.markdown('<div class="tip">💡 Have many listings? Upload a CSV and get a fair price and a fraud verdict for every row.</div>',
                unsafe_allow_html=True)
    tmpl = df[FEATURES].head(5).copy()
    tmpl["asking_price"] = [round(float(np.exp(v)) * rate) for v in df[TARGET].head(5)]
    st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False).encode("utf-8"),
                       "resaleradar_template.csv", "text/csv")
    st.caption(f"Keep the same column names and values as the template. The optional **asking_price** column "
               f"should be in {sym}. Without it you only get fair prices.")

    up = st.file_uploader("Upload your CSV", type=["csv"])
    if up is not None:
        try:
            user_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read that file: {e}")
            user_df = None
        if user_df is not None:
            missing = [c for c in FEATURES if c not in user_df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}. Please use the template.")
            else:
                work = user_df.copy()
                for c in NUMS:
                    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(df[c].median())
                for c in CATS:
                    work[c] = work[c].astype(str).replace("nan", str(df[c].mode()[0]))
                X_b = encode(work, cols)
                pred_b = models[model_name].predict(X_b)

                out = user_df.copy()
                out["fair_price"] = np.round(np.exp(pred_b) * rate)
                if "asking_price" in user_df.columns:
                    ask = pd.to_numeric(user_df["asking_price"], errors="coerce")
                    valid = (ask.notna() & (ask > 0)).values
                    fraud_col = np.full(len(out), np.nan)
                    vs_col = np.full(len(out), np.nan)
                    verdict_col = np.array(["-"] * len(out), dtype=object)
                    if valid.any():
                        a_norm = np.log(ask.values[valid] / rate)
                        sc_b, ra_b, _ = fraud_scores(X_b.loc[valid], a_norm, pred_b[valid])
                        fraud_col[valid] = np.round(sc_b * 100)
                        vs_col[valid] = np.round((ra_b - 1) * 100)
                        verdict_col[valid] = [verdict_for(r_, s_)[0] for r_, s_ in zip(ra_b, sc_b)]
                    out["vs_fair_percent"] = vs_col
                    out["fraud_score"] = fraud_col
                    out["verdict"] = verdict_col

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Listings checked", len(out))
                    m2.metric("🚨 Suspicious", int((verdict_col == "🚨 SUSPICIOUS").sum()))
                    m3.metric("⚠️ Caution", int((verdict_col == "⚠️ CAUTION").sum()))
                    m4.metric("💸 Overpriced", int((verdict_col == "💸 OVERPRICED").sum()))
                else:
                    st.metric("Listings checked", len(out))

                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download results", out.to_csv(index=False).encode("utf-8"),
                                   "resaleradar_results.csv", "text/csv")

# ---------------------------------------------------------------- MARKET EXPLORER
with t_market:
    st.markdown('<div class="tip">💡 Explore what drives prices in the market. Hover over charts for details.</div>',
                unsafe_allow_html=True)
    template = "plotly_dark"
    a, b = st.columns(2)
    with a:
        if "device_brand" in df.columns:
            top = df["device_brand"].value_counts().head(12).index
            fig = px.box(df[df["device_brand"].isin(top)], x="device_brand", y=TARGET, color="device_brand",
                         template=template, title="Price by brand (top 12)")
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
                             title="Days used vs price")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with d:
        corr = df[NUMS + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis", template=template,
                        title="Which specs move with price?")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- MODEL PERFORMANCE
with t_perf:
    st.markdown('<div class="tip">💡 How accurate are the models? R² closer to 1 is better. '
                'MAE and RMSE are average errors, so lower is better.</div>', unsafe_allow_html=True)
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
        fig = px.bar(imp, orientation="h", template="plotly_dark", title="Top 10 price drivers (Random Forest)")
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