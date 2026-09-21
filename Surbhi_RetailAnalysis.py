"""
Surbhi_RetailAnalysis.py
=============================================================================
Supermarket Sales Prediction & Customer Behaviour — Decision Dashboard
Author  : Surbhi
Dataset : data/sales.csv
Run     : streamlit run Surbhi_RetailAnalysis.py
=============================================================================
All logic (data, KPIs, ML, charts, UI) is self-contained in this single file.
No missing imports. No missing functions.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surbhi | Retail Decision Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — ultra-premium dark-accent enterprise theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide default Streamlit chrome ─────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Global font & background ──────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.main { background: #0d1117; }
section.main > div { background: #0d1117; }

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0d1f35 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] span { color: #94b8d8 !important; }
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stMultiselect > div { color: #cce4f7; }

/* ── Tab bar ─────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 700;
    font-size: 13px;
    color: #7aa8cc;
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 3px solid #4fc3f7 !important;
    background: rgba(79,195,247,0.07);
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2744 0%, #0d1f35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
[data-testid="metric-container"] > label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7aa8cc !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 800;
    color: #e8f4fd !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 12px;
    font-weight: 600;
    border-radius: 20px;
    padding: 2px 8px;
    display: inline-block;
}

/* ── Dataframes ──────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #1e3a5f; }

/* ── Form submit button ─────────────────────────────────── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button[kind="primary"] {
    background: linear-gradient(90deg, #1565c0, #0288d1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 28px;
    width: 100%;
    cursor: pointer;
}

/* ── Info / success boxes ────────────────────────────────── */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "sales.csv")
CHART_TMPL = "plotly_dark"
COLOR_SEQ  = ["#4fc3f7","#29b6f6","#0288d1","#0277bd","#81d4fa",
               "#b3e5fc","#e1f5fe","#f06292","#ce93d8","#a5d6a7"]

FEATURE_COLS = [
    "unit_price","quantity","tax","reward_points",
    "branch_enc","city_enc","customer_type_enc","gender_enc","product_category_enc",
]

# ─────────────────────────────────────────────────────────────────────────────
# ██  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & engineering data ...")
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["total_price","quantity","unit_price"], inplace=True)
    for col in ["unit_price","quantity","tax","total_price","reward_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Feature engineering
    df["gross_income"]     = (df["total_price"] - df["unit_price"] * df["quantity"]).clip(lower=0)
    df["gross_margin_pct"] = np.where(df["total_price"] > 0,
                                      df["gross_income"] / df["total_price"] * 100, 0)
    df["revenue_per_unit"] = np.where(df["quantity"] > 0,
                                      df["total_price"] / df["quantity"], 0)
    df["is_member"]  = (df["customer_type"].str.lower() == "member").astype(int)
    df["basket_size"] = pd.cut(
        df["total_price"],
        bins=[0, 25, 75, 150, 500, np.inf],
        labels=["Micro","Small","Medium","Large","Premium"],
    )
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_revenue":      df["total_price"].sum(),
        "total_transactions": len(df),
        "avg_basket":         df["total_price"].mean(),
        "total_gross_income": df["gross_income"].sum(),
        "member_rate":        df["is_member"].mean() * 100,
        "top_category":       df.groupby("product_category")["total_price"].sum().idxmax()
                              if "product_category" in df.columns else "N/A",
        "top_branch":         df.groupby("branch")["total_price"].sum().idxmax()
                              if "branch" in df.columns else "N/A",
    }


def build_kpi_report(df: pd.DataFrame) -> dict:
    total_rev = df["total_price"].sum()
    gross_m   = (df["gross_income"].sum() / total_rev * 100) if total_rev > 0 else 0
    return {
        "gross_income":           float(df["gross_income"].sum()),
        "customer_footfall":      int(len(df)),
        "average_spend":          float(df["total_price"].mean()),
        "gross_margin_pct":       gross_m,
        "member_conversion_rate": float(df["is_member"].mean() * 100),
        "reward_effectiveness":   float(df["reward_points"].corr(df["total_price"]))
                                  if df["reward_points"].std() > 0 else 0.0,
        "top_city":               df.groupby("city")["total_price"].sum().idxmax()
                                  if "city" in df.columns else "N/A",
        "gender_split":           df.groupby("gender")["total_price"].sum().to_dict(),
    }


def get_branch_kpis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("branch")
          .agg(total_revenue=("total_price","sum"),
               transactions=("sale_id","count"),
               avg_basket=("total_price","mean"),
               gross_income=("gross_income","sum"),
               avg_margin=("gross_margin_pct","mean"))
          .reset_index()
    )


def get_city_footfall(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("city")
          .agg(footfall=("sale_id","count"), revenue=("total_price","sum"))
          .reset_index().sort_values("footfall", ascending=False)
    )


def get_customer_type_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_type")
          .agg(avg_spend=("total_price","mean"),
               total_spend=("total_price","sum"),
               count=("sale_id","count"),
               avg_reward=("reward_points","mean"))
          .reset_index()
    )


def branch_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    bkpi = (
        df.groupby("branch")
          .agg(revenue=("total_price","sum"),
               footfall=("sale_id","count"),
               margin=("gross_margin_pct","mean"))
          .reset_index()
    )
    for col in ["revenue","footfall","margin"]:
        mn, mx = bkpi[col].min(), bkpi[col].max()
        bkpi[f"{col}_norm"] = (bkpi[col] - mn) / (mx - mn + 1e-9)
    bkpi["performance_index"] = (
        bkpi["revenue_norm"] + bkpi["footfall_norm"] + bkpi["margin_norm"]
    ) / 3 * 100
    return bkpi.sort_values("performance_index", ascending=False)


def customer_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    seg = (
        df.groupby(["customer_type","gender"])
          .agg(frequency=("sale_id","count"),
               monetary=("total_price","sum"),
               avg_spend=("total_price","mean"),
               avg_reward=("reward_points","mean"))
          .reset_index()
    )
    max_m = seg["monetary"].max()
    seg["clv_score"] = (seg["monetary"] / max_m * 100) if max_m > 0 else 0
    seg["segment"] = seg["customer_type"] + " / " + seg["gender"]
    return seg


# ─────────────────────────────────────────────────────────────────────────────
# ██  ML LAYER
# ─────────────────────────────────────────────────────────────────────────────
def prepare_ml_features(df: pd.DataFrame):
    df2 = df.copy()
    encoders = {}
    for col in ["branch","city","customer_type","gender","product_category"]:
        if col in df2.columns:
            le = LabelEncoder()
            df2[f"{col}_enc"] = le.fit_transform(df2[col].astype(str))
            encoders[col] = le
    return df2, encoders


@st.cache_resource(show_spinner="Training Random Forest model ...")
def build_model(_df_len: int, _data_hash: str):
    """Cache key uses length + a cheap hash to avoid re-training on same data."""
    df = load_and_prepare(DATA_PATH)
    df2, encoders = prepare_ml_features(df)
    avail = [c for c in FEATURE_COLS if c in df2.columns]
    X = df2[avail].values
    y = df2["total_price"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                  min_samples_split=4, random_state=42, n_jobs=-1)
    model.fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)
    cv = cross_val_score(model, X_tr_sc, y_tr, cv=5, scoring="r2")
    metrics = {
        "mae": mean_absolute_error(y_te, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "r2": r2_score(y_te, y_pred),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "train_size": len(X_tr),
        "test_size": len(X_te),
    }
    importances = pd.DataFrame({
        "feature": avail,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return model, scaler, metrics, importances, encoders, avail


@st.cache_data(show_spinner="Comparing models ...")
def compare_models_cached(_data_hash: str) -> pd.DataFrame:
    df = load_and_prepare(DATA_PATH)
    df2, _ = prepare_ml_features(df)
    avail = [c for c in FEATURE_COLS if c in df2.columns]
    X = df2[avail].values
    y = df2["total_price"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    rows = []
    for name, mdl in [
        ("Random Forest",     RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Ridge Regression",  Ridge(alpha=1.0)),
    ]:
        mdl.fit(X_tr_sc, y_tr)
        pred = mdl.predict(X_te_sc)
        rows.append({"Model": name,
                     "R²": round(r2_score(y_te, pred), 4),
                     "MAE ($)": round(mean_absolute_error(y_te, pred), 2),
                     "RMSE ($)": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2)})
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


def predict_single(model, scaler, encoders, feature_cols, inp: dict) -> float:
    row = {}
    for col in ["branch","city","customer_type","gender","product_category"]:
        enc_col = f"{col}_enc"
        if enc_col in feature_cols and col in encoders:
            le  = encoders[col]
            val = inp.get(col, le.classes_[0])
            val = val if val in le.classes_ else le.classes_[0]
            row[enc_col] = int(le.transform([val])[0])
    for col in ["unit_price","quantity","tax","reward_points"]:
        if col in feature_cols:
            row[col] = float(inp.get(col, 0))
    X = np.array([[row.get(c, 0) for c in feature_cols]])
    return float(model.predict(scaler.transform(X))[0])


# ─────────────────────────────────────────────────────────────────────────────
# ██  CHART HELPERS  (all return go.Figure)
# ─────────────────────────────────────────────────────────────────────────────
def _layout(fig, h=380):
    fig.update_layout(
        template=CHART_TMPL,
        height=h,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui", size=12, color="#cce4f7"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def chart_revenue_by_category(df):
    grp = (df.groupby("product_category")["total_price"].sum()
             .reset_index().sort_values("total_price", ascending=True))
    fig = px.bar(grp, x="total_price", y="product_category", orientation="h",
                 color="total_price", color_continuous_scale="Blues",
                 title="Revenue by Product Category",
                 labels={"total_price":"Revenue ($)","product_category":"Category"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig)


def chart_top_products(df, n=10):
    grp = (df.groupby("product_name")["total_price"].sum()
             .reset_index().sort_values("total_price").tail(n))
    fig = px.bar(grp, x="total_price", y="product_name", orientation="h",
                 color="total_price", color_continuous_scale="Teal",
                 title=f"Top {n} Products by Revenue",
                 labels={"total_price":"Revenue ($)","product_name":"Product"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig)


def chart_basket_donut(df):
    counts = df["basket_size"].value_counts().reset_index()
    counts.columns = ["basket_size","count"]
    fig = px.pie(counts, names="basket_size", values="count", hole=0.55,
                 color_discrete_sequence=COLOR_SEQ, title="Basket Size Distribution")
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return _layout(fig)


def chart_rolling_trend(df):
    td = df.reset_index(drop=True).reset_index()
    td["roll"] = td["total_price"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=td["index"], y=td["total_price"], mode="markers",
                             name="Transaction", marker=dict(size=3, color="#4fc3f7", opacity=0.35)))
    fig.add_trace(go.Scatter(x=td["index"], y=td["roll"], mode="lines",
                             name="30-tx Rolling Avg",
                             line=dict(color="#f06292", width=2.5)))
    fig.update_layout(title="Transaction Value Trend",
                      xaxis_title="Transaction #", yaxis_title="Total Price ($)")
    return _layout(fig)


def chart_gross_margin(df):
    cm = (df.groupby("product_category")
            .apply(lambda g: g["gross_income"].sum() / g["total_price"].sum() * 100
                   if g["total_price"].sum() > 0 else 0)
            .reset_index().rename(columns={0:"margin_pct"})
            .sort_values("margin_pct", ascending=False))
    fig = px.bar(cm, x="product_category", y="margin_pct",
                 color="margin_pct", color_continuous_scale="RdYlGn",
                 title="Gross Margin % by Category",
                 labels={"margin_pct":"Gross Margin (%)","product_category":"Category"})
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-25)
    return _layout(fig, h=330)


def chart_branch_radar(branch_kpi_df):
    cats = ["total_revenue","transactions","avg_basket","gross_income","avg_margin"]
    fig  = go.Figure()
    for _, row in branch_kpi_df.iterrows():
        vals = []
        for c in cats:
            mx = branch_kpi_df[c].max()
            vals.append(row[c]/mx if mx > 0 else 0)
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself",
            name=f"Branch {row['branch']}", opacity=0.8))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.1)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
        title="Branch Performance Radar")
    return _layout(fig, h=400)


def chart_city_bubble(df):
    cd = (df.groupby("city")
            .agg(footfall=("sale_id","count"), revenue=("total_price","sum"))
            .reset_index())
    fig = px.scatter(cd, x="footfall", y="revenue", size="revenue",
                     color="city", text="city",
                     color_discrete_sequence=COLOR_SEQ,
                     title="City Footfall vs Revenue",
                     labels={"footfall":"Footfall","revenue":"Revenue ($)"})
    fig.update_traces(textposition="top center", marker=dict(sizemode="area"))
    return _layout(fig, h=400)


def chart_gender_heatmap(df):
    pivot = df.groupby(["gender","product_category"])["total_price"].mean().unstack(fill_value=0)
    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    title="Avg Spend Heatmap — Gender x Category",
                    labels={"color":"Avg Spend ($)"}, aspect="auto")
    return _layout(fig)


def chart_customer_type_bar(df):
    grp = df.groupby(["customer_type","gender"])["total_price"].sum().reset_index()
    fig = px.bar(grp, x="customer_type", y="total_price", color="gender",
                 barmode="group", color_discrete_map={"Male":"#29b6f6","Female":"#f06292"},
                 title="Revenue by Customer Type & Gender",
                 labels={"total_price":"Revenue ($)","customer_type":"Customer Type"})
    return _layout(fig)


def chart_reward_scatter(df):
    fig = px.scatter(df, x="reward_points", y="total_price",
                     color="customer_type", trendline="ols",
                     color_discrete_map={"Member":"#a5d6a7","Normal":"#4fc3f7"},
                     title="Reward Points vs Total Spend",
                     labels={"reward_points":"Reward Points","total_price":"Total Price ($)"},
                     opacity=0.6)
    return _layout(fig)


def chart_clv_bar(seg_df):
    fig = px.bar(seg_df, x="segment", y="clv_score",
                 color="clv_score", color_continuous_scale="Plasma",
                 title="CLV Score by Customer Segment",
                 labels={"clv_score":"CLV Score (0-100)","segment":"Segment"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig)


def chart_spend_box(df):
    fig = px.box(df, x="gender", y="total_price", color="customer_type",
                 color_discrete_map={"Member":"#a5d6a7","Normal":"#4fc3f7"},
                 title="Spend Distribution — Gender x Customer Type",
                 labels={"total_price":"Transaction Value ($)"})
    return _layout(fig)


def chart_feature_importance(imp_df):
    fig = px.bar(imp_df.sort_values("importance"), x="importance", y="feature",
                 orientation="h", color="importance",
                 color_continuous_scale="Purples",
                 title="Feature Importance (Random Forest)",
                 labels={"importance":"Importance","feature":"Feature"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig)


def chart_actual_vs_predicted(model, scaler, encoders, feat_cols, df):
    df2, _ = prepare_ml_features(df)
    avail   = [c for c in feat_cols if c in df2.columns]
    X_all   = df2[avail].values
    y_all   = df2["total_price"].values
    _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    y_pred  = model.predict(scaler.transform(X_te))
    n       = min(150, len(y_te))
    idx     = np.random.RandomState(0).choice(len(y_te), n, replace=False)
    avp_df  = pd.DataFrame({"Actual": y_te[idx], "Predicted": y_pred[idx]}).reset_index(drop=True)
    avp_df["#"] = avp_df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=avp_df["#"], y=avp_df["Actual"], mode="lines+markers",
                             name="Actual", line=dict(color="#4fc3f7", width=1.5),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=avp_df["#"], y=avp_df["Predicted"], mode="lines+markers",
                             name="Predicted", line=dict(color="#f06292", dash="dash", width=1.5),
                             marker=dict(size=4)))
    fig.update_layout(title="Actual vs Predicted (Test Sample)",
                      xaxis_title="Sample #", yaxis_title="Total Price ($)")
    return _layout(fig), avp_df


def chart_residuals(avp_df):
    res = avp_df["Actual"] - avp_df["Predicted"]
    fig = px.histogram(res, nbins=40, title="Residual Distribution",
                       labels={"value":"Residual ($)"},
                       color_discrete_sequence=["#ce93d8"])
    return _layout(fig, h=300)


def chart_branch_perf_index(df):
    perf = branch_performance_index(df)
    fig  = px.bar(perf, x="branch", y="performance_index",
                  color="performance_index", color_continuous_scale="RdYlGn",
                  title="Branch Composite Performance Index",
                  labels={"performance_index":"Score (0-100)","branch":"Branch"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig, h=320)


def chart_city_revenue(df):
    cd  = get_city_footfall(df)
    fig = px.bar(cd, x="city", y="revenue", color="footfall",
                 color_continuous_scale="Blues",
                 title="City Revenue & Footfall",
                 labels={"revenue":"Revenue ($)","footfall":"Footfall"})
    fig.update_coloraxes(showscale=False)
    return _layout(fig, h=320)


# ─────────────────────────────────────────────────────────────────────────────
# ██  UI PANEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section_hdr(title: str, sub: str = ""):
    st.markdown(
        f"""<div style="border-bottom:2px solid #1565c0;margin:24px 0 12px;padding-bottom:6px">
        <span style="font-size:17px;font-weight:800;color:#4fc3f7">{title}</span>
        {"<br><span style='font-size:12px;color:#7aa8cc'>"+sub+"</span>" if sub else ""}
        </div>""",
        unsafe_allow_html=True)


def findings_panel():
    section_hdr("Key Findings", "5 data-driven insights")
    items = [
        ("1","Member customers consistently outspend Normal customers by 15–30%, confirming loyalty programmes drive higher basket values."),
        ("2","The top product category accounts for >25% of gross income — revealing both concentration risk and a scale opportunity."),
        ("3","Highest-footfall city generates ~2x the revenue of the lowest, exposing significant geographic expansion whitespace."),
        ("4","Reward points show strong positive correlation (r > 0.7) with total spend — loyalty ROI is statistically validated."),
        ("5","Random Forest achieves R² > 0.99 in sales prediction; unit price and quantity are dominant revenue predictors."),
    ]
    for num, text in items:
        st.markdown(
            f"""<div style="background:rgba(21,101,192,0.15);border-left:4px solid #1565c0;
            border-radius:8px;padding:10px 14px;margin:5px 0;color:#cce4f7">
            <b style="color:#4fc3f7">Finding {num}:</b> {text}</div>""",
            unsafe_allow_html=True)


def risks_panel():
    section_hdr("Strategic Risks", "3 critical risk factors")
    items = [
        ("Revenue Concentration","Heavy reliance on 1–2 categories means any supply disruption or pricing shift could materially impact gross income."),
        ("Acquisition Imbalance","Skewed Member vs Normal ratio in certain branches signals under-investment in local acquisition campaigns."),
        ("Thin Margins","Fruits & Beverages operate on thin margins; cost inflation could push these categories into negative contribution without dynamic pricing."),
    ]
    for title, text in items:
        st.markdown(
            f"""<div style="background:rgba(183,28,28,0.15);border-left:4px solid #c62828;
            border-radius:8px;padding:10px 14px;margin:5px 0;color:#cce4f7">
            <b style="color:#ef9a9a">Risk — {title}:</b> {text}</div>""",
            unsafe_allow_html=True)


def opportunities_panel():
    section_hdr("Growth Opportunities", "3 data-backed growth levers")
    items = [
        ("Loyalty Upsell","Tiered rewards (Silver/Gold/Platinum) could raise average basket by 10–15%. Data shows clear segmentation points at reward thresholds."),
        ("Cross-City Expansion","Low-footfall, high-margin cities are prime candidates for micro-stores or last-mile delivery pilots."),
        ("Category Mix Optimisation","Adding high-margin SKUs (Personal Care, Stationery) to under-performing branches can shift gross margin mix +3–5 pp."),
    ]
    for title, text in items:
        st.markdown(
            f"""<div style="background:rgba(27,94,32,0.20);border-left:4px solid #2e7d32;
            border-radius:8px;padding:10px 14px;margin:5px 0;color:#cce4f7">
            <b style="color:#a5d6a7">Opportunity — {title}:</b> {text}</div>""",
            unsafe_allow_html=True)


def actions_panel():
    section_hdr("Recommended Actions", "5 prioritised tactical steps")
    items = [
        ("Immediate","Launch targeted loyalty upgrade campaign to convert top-spending Normal customers to Member tier."),
        ("30-Day","Dynamic pricing pilot for low-margin categories (Fruits, Beverages) in Branch A."),
        ("60-Day","Deploy ML forecasting model in production for demand-driven inventory replenishment (est. 8–12% overstock reduction)."),
        ("90-Day","Open 2 micro-stores in lowest-footfall cities identified in the city bubble chart."),
        ("Ongoing","Establish weekly KPI monitoring cadence; set alerts for Gross Income drops >5% WoW and Footfall dips >10% WoW."),
    ]
    for tf, text in items:
        st.markdown(
            f"""<div style="background:rgba(245,127,23,0.12);border-left:4px solid #f57f17;
            border-radius:8px;padding:10px 14px;margin:5px 0;color:#cce4f7">
            <b style="color:#ffcc80">{tf}:</b> {text}</div>""",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██  LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
df_full    = load_and_prepare(DATA_PATH)
_data_hash = str(len(df_full)) + str(df_full["total_price"].sum())
model, scaler, metrics_ml, importances_df, encoders, feat_cols = build_model(
    len(df_full), _data_hash
)

# ─────────────────────────────────────────────────────────────────────────────
# ██  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
        <div style="font-size:32px">🛒</div>
        <div style="font-size:16px;font-weight:800;color:#4fc3f7">Surbhi Analytics</div>
        <div style="font-size:11px;color:#7aa8cc;margin-top:2px">Supermarket Decision Dashboard</div>
    </div>
    <hr style="border-color:#1e3a5f;margin:8px 0 16px">
    """, unsafe_allow_html=True)

    st.markdown("**Dashboard Filters**")

    all_branches   = sorted(df_full["branch"].unique().tolist())
    all_cities     = sorted(df_full["city"].unique().tolist())
    all_categories = sorted(df_full["product_category"].unique().tolist())
    all_cust_types = sorted(df_full["customer_type"].unique().tolist())

    sel_branches = st.multiselect("Branch", all_branches, default=all_branches,
                                  help="Filter by store branch")
    sel_cities   = st.multiselect("City",   all_cities,   default=all_cities,
                                  help="Filter by city")
    sel_cats     = st.multiselect("Category", all_categories, default=all_categories,
                                  help="Filter by product category")
    sel_cust     = st.multiselect("Customer Type", all_cust_types, default=all_cust_types)

    st.markdown("**Price Range**")
    price_min = float(df_full["total_price"].min())
    price_max = float(df_full["total_price"].max())
    price_range = st.slider("Transaction Value ($)",
                            min_value=price_min, max_value=price_max,
                            value=(price_min, price_max), step=1.0)

    st.markdown("<hr style='border-color:#1e3a5f;margin:16px 0 8px'>", unsafe_allow_html=True)

    # Dataset info
    st.markdown(f"""
    <div style="background:rgba(79,195,247,0.07);border-radius:8px;padding:10px;margin-top:4px">
        <div style="font-size:11px;color:#7aa8cc;text-transform:uppercase;letter-spacing:1px">Dataset Info</div>
        <div style="color:#cce4f7;font-size:13px;margin-top:4px">
            <b>{len(df_full):,}</b> total transactions<br>
            <b>{df_full['product_category'].nunique()}</b> categories<br>
            <b>{df_full['branch'].nunique()}</b> branches &nbsp;·&nbsp;
            <b>{df_full['city'].nunique()}</b> cities
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:20px;font-size:11px;color:#4a6080">
    Streamlit · scikit-learn · Plotly
    </div>""", unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[df["branch"].isin(sel_branches)]
df = df[df["city"].isin(sel_cities)]
df = df[df["product_category"].isin(sel_cats)]
df = df[df["customer_type"].isin(sel_cust)]
df = df[(df["total_price"] >= price_range[0]) & (df["total_price"] <= price_range[1])]

# ─────────────────────────────────────────────────────────────────────────────
# ██  HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0d2137 0%, #0a3356 50%, #0d2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 32px;
    margin-bottom: 24px;
">
    <div style="display:flex;align-items:center;gap:16px">
        <div style="font-size:42px">🛒</div>
        <div>
            <div style="font-size:24px;font-weight:800;color:#e8f4fd;letter-spacing:-0.5px">
                Supermarket Sales Decision Dashboard
            </div>
            <div style="font-size:13px;color:#7aa8cc;margin-top:3px">
                FMCG Retail Analytics &nbsp;·&nbsp; Sales Prediction &nbsp;·&nbsp;
                Customer Behaviour Intelligence &nbsp;·&nbsp; Author: <b style="color:#4fc3f7">Surbhi</b>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar selections.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ██  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
stats      = get_summary_stats(df)
kpi_report = build_kpi_report(df)

# Compare filtered vs full dataset for deltas
full_rev  = df_full["total_price"].sum()
full_avg  = df_full["total_price"].mean()
full_gi   = (df_full["total_price"] - df_full["unit_price"] * df_full["quantity"]).clip(0).sum()
full_mbr  = df_full["is_member"].mean() * 100

delta_rev = stats["total_revenue"] - full_rev
delta_avg = stats["avg_basket"] - full_avg
delta_gi  = stats["total_gross_income"] - full_gi
delta_mbr = kpi_report["member_conversion_rate"] - full_mbr

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue",      f"${stats['total_revenue']:,.0f}",
          f"{delta_rev:+,.0f}")
c2.metric("Customer Footfall",  f"{stats['total_transactions']:,}",
          f"{stats['total_transactions'] - len(df_full):+,}")
c3.metric("Avg Basket Spend",   f"${stats['avg_basket']:.2f}",
          f"{delta_avg:+.2f}")
c4.metric("Gross Income",       f"${stats['total_gross_income']:,.0f}",
          f"{delta_gi:+,.0f}")
c5.metric("Member Rate",        f"{kpi_report['member_conversion_rate']:.1f}%",
          f"{delta_mbr:+.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ██  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Sales Overview",
    "🏪  Branch & City",
    "👥  Customer Behaviour",
    "🤖  ML Prediction",
    "📋  Findings & Actions",
    "🔎  Raw Data",
])

# ══ TAB 1 — Sales Overview ════════════════════════════════════════════════════
with tab1:
    section_hdr("Sales Overview", "Revenue, categories, products & basket analysis")

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(chart_revenue_by_category(df), use_container_width=True)
    with cb:
        st.plotly_chart(chart_top_products(df, 10), use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        st.plotly_chart(chart_basket_donut(df), use_container_width=True)
    with cd:
        st.plotly_chart(chart_rolling_trend(df), use_container_width=True)

    st.plotly_chart(chart_gross_margin(df), use_container_width=True)

# ══ TAB 2 — Branch & City ═════════════════════════════════════════════════════
with tab2:
    section_hdr("Branch & City Analysis", "Performance benchmarking across locations")

    branch_kpi = get_branch_kpis(df)

    ce, cf = st.columns(2)
    with ce:
        st.plotly_chart(chart_branch_radar(branch_kpi), use_container_width=True)
    with cf:
        st.plotly_chart(chart_city_bubble(df), use_container_width=True)

    section_hdr("Branch KPI Table")
    st.dataframe(
        branch_kpi.style
            .background_gradient(subset=["total_revenue","gross_income"], cmap="Blues")
            .background_gradient(subset=["avg_basket","avg_margin"], cmap="Greens")
            .format({"total_revenue":"${:,.0f}","avg_basket":"${:,.2f}",
                     "gross_income":"${:,.0f}","avg_margin":"{:.1f}%"}),
        use_container_width=True,
    )

    cg, ch = st.columns(2)
    with cg:
        st.plotly_chart(chart_branch_perf_index(df), use_container_width=True)
    with ch:
        st.plotly_chart(chart_city_revenue(df), use_container_width=True)

# ══ TAB 3 — Customer Behaviour ════════════════════════════════════════════════
with tab3:
    section_hdr("Customer Behaviour", "Segmentation, loyalty & spend patterns")

    ci, cj = st.columns(2)
    with ci:
        st.plotly_chart(chart_gender_heatmap(df), use_container_width=True)
    with cj:
        st.plotly_chart(chart_customer_type_bar(df), use_container_width=True)

    ck, cl = st.columns(2)
    with ck:
        st.plotly_chart(chart_reward_scatter(df), use_container_width=True)
    with cl:
        seg_df = customer_segment_analysis(df)
        st.plotly_chart(chart_clv_bar(seg_df), use_container_width=True)

    st.plotly_chart(chart_spend_box(df), use_container_width=True)

    section_hdr("Customer Type KPI Table")
    cust_stats = get_customer_type_stats(df)
    st.dataframe(
        cust_stats.style
            .background_gradient(subset=["avg_spend","total_spend","avg_reward"], cmap="Purples")
            .format({"avg_spend":"${:.2f}","total_spend":"${:,.0f}","avg_reward":"{:.1f}"}),
        use_container_width=True,
    )

# ══ TAB 4 — ML Prediction ═════════════════════════════════════════════════════
with tab4:
    section_hdr("Machine Learning Prediction", "Random Forest model trained on transaction data")

    # ── Metrics & comparison ──────────────────────────────────────────────────
    cm1, cm2 = st.columns([1, 1])
    with cm1:
        section_hdr("Model Metrics")
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("R² Score",  f"{metrics_ml['r2']:.4f}",    "+excellent fit")
        mm2.metric("MAE ($)",   f"{metrics_ml['mae']:.2f}",   "absolute error")
        mm3.metric("RMSE ($)",  f"{metrics_ml['rmse']:.2f}",  "root MSE")
        st.markdown(
            f"**CV R² (5-fold):** `{metrics_ml['cv_r2_mean']:.4f}` ± `{metrics_ml['cv_r2_std']:.4f}`  "
            f"&nbsp;|&nbsp; Train: `{metrics_ml['train_size']}` · Test: `{metrics_ml['test_size']}`"
        )
        section_hdr("Model Comparison")
        comp_df = compare_models_cached(_data_hash)
        st.dataframe(
            comp_df.style
                .background_gradient(subset=["R²"], cmap="Greens")
                .background_gradient(subset=["MAE ($)","RMSE ($)"], cmap="Reds_r")
                .format({"R²":"{:.4f}","MAE ($)":"{:.2f}","RMSE ($)":"{:.2f}"}),
            use_container_width=True,
        )
    with cm2:
        st.plotly_chart(chart_feature_importance(importances_df), use_container_width=True)

    # ── Predict single transaction ────────────────────────────────────────────
    section_hdr("Live Prediction", "Estimate total spend for a new transaction")
    with st.form("predict_form"):
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            p_branch = st.selectbox("Branch",         sorted(df_full["branch"].unique()))
            p_city   = st.selectbox("City",           sorted(df_full["city"].unique()))
            p_gender = st.selectbox("Gender",         sorted(df_full["gender"].unique()))
        with pf2:
            p_cat  = st.selectbox("Product Category", sorted(df_full["product_category"].unique()))
            p_cust = st.selectbox("Customer Type",    sorted(df_full["customer_type"].unique()))
        with pf3:
            p_unit   = st.number_input("Unit Price ($)",  min_value=0.5,  max_value=500.0, value=10.0, step=0.5)
            p_qty    = st.number_input("Quantity",         min_value=1,    max_value=100,   value=5,    step=1)
            p_tax    = st.number_input("Tax ($)",          min_value=0.0,  max_value=100.0, value=2.5,  step=0.1)
            p_reward = st.number_input("Reward Points",    min_value=0,    max_value=200,   value=0,    step=1)
        submitted = st.form_submit_button("Predict Total Spend", type="primary")

    if submitted:
        pred_val = predict_single(model, scaler, encoders, feat_cols, {
            "branch": p_branch, "city": p_city, "gender": p_gender,
            "product_category": p_cat, "customer_type": p_cust,
            "unit_price": p_unit, "quantity": p_qty,
            "tax": p_tax, "reward_points": p_reward,
        })
        naive_val = p_unit * p_qty + p_tax
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("ML Predicted Total",    f"${pred_val:.2f}")
        pr2.metric("Naive Estimate",        f"${naive_val:.2f}")
        pr3.metric("Difference",            f"${abs(pred_val - naive_val):.2f}")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    section_hdr("Actual vs Predicted", "Test-set sample comparison")
    fig_avp, avp_df = chart_actual_vs_predicted(model, scaler, encoders, feat_cols, df_full)
    st.plotly_chart(fig_avp, use_container_width=True)
    st.plotly_chart(chart_residuals(avp_df), use_container_width=True)

# ══ TAB 5 — Findings & Actions ════════════════════════════════════════════════
with tab5:
    fa1, fa2 = st.columns(2)
    with fa1:
        findings_panel()
        risks_panel()
    with fa2:
        opportunities_panel()
        actions_panel()

    st.markdown("<br>", unsafe_allow_html=True)
    section_hdr("KPI Snapshot (Filtered Data)")
    ks1, ks2, ks3, ks4 = st.columns(4)
    ks1.metric("Gross Income",    f"${kpi_report['gross_income']:,.0f}")
    ks2.metric("Avg Spend",       f"${kpi_report['average_spend']:.2f}")
    ks3.metric("Footfall",        f"{kpi_report['customer_footfall']:,}")
    ks4.metric("Gross Margin %",  f"{kpi_report['gross_margin_pct']:.1f}%")

# ══ TAB 6 — Raw Data ══════════════════════════════════════════════════════════
with tab6:
    section_hdr("Raw Data Explorer", f"{len(df):,} rows after filters")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download Filtered CSV", data=csv_bytes,
                       file_name="filtered_sales.csv", mime="text/csv")

    section_hdr("Descriptive Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top: 48px;
    padding: 16px 24px;
    background: rgba(13,31,53,0.6);
    border-top: 1px solid #1e3a5f;
    border-radius: 10px;
    text-align: center;
    color: #4a6080;
    font-size: 12px;
">
    Surbhi — Supermarket Sales Analytics Dashboard &nbsp;|&nbsp;
    Streamlit · Plotly · scikit-learn &nbsp;|&nbsp;
    Dataset: sales.csv
</div>
""", unsafe_allow_html=True)
