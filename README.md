# 🛒 Supermarket Sales Prediction & Customer Behaviour Dashboard

**Author:** Surbhi  
**Dataset:** `data/sales.csv`  
**Stack:** Python · Streamlit · scikit-learn · Plotly  

---

## 📁 Project Structure

```
surbhi/
├── Surbhi_RetailAnalysis.py     ← Main Streamlit dashboard (run this)
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
├── Surbhi_ProjectReport.docx    ← Comprehensive Word report
│
├── backend/
│   ├── data_processor.py        ← Data loading, cleaning & feature engineering
│   ├── kpi_engine.py            ← FMCG KPI calculations
│   └── ml_model.py              ← Random Forest prediction pipeline
│
├── data/
│   └── sales.csv                ← Raw dataset (1 000+ supermarket transactions)
│
├── frontend/
│   └── components.py            ← Reusable Plotly/Streamlit UI components
│
├── model/
│   ├── train_model.py           ← Standalone model training script
│   ├── rf_model.pkl             ← Saved Random Forest model (generated)
│   ├── scaler.pkl               ← Saved StandardScaler (generated)
│   ├── encoders.pkl             ← Saved LabelEncoders (generated)
│   └── model_meta.json          ← Model metrics & feature importances (generated)
│
└── report_images/               ← Auto-generated chart PNGs for the Word report
```

---

## 🚀 Quick Start

### 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 2 — (Optional) Pre-train the Model

```bash
python model/train_model.py
```

This saves `rf_model.pkl`, `scaler.pkl`, `encoders.pkl`, and `model_meta.json` to the `model/` folder. If you skip this step the dashboard trains the model on first load (cached via `@st.cache_resource`).

### 3 — Launch the Dashboard

```bash
streamlit run Surbhi_RetailAnalysis.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Sales Overview** | Revenue by category, top products, basket size distribution, rolling revenue trend, gross margin % |
| **Branch & City** | Radar comparison, city footfall bubble chart, branch KPI table, performance index |
| **Customer Behaviour** | Gender × category heatmap, loyalty analysis, reward-points scatter, CLV segmentation |
| **ML Prediction** | Model metrics (R², MAE, RMSE), feature importances, model comparison, live prediction form, actual vs predicted chart |
| **Findings & Actions** | 5 Key Findings · 3 Risks · 3 Opportunities · 5 Recommended Actions |
| **Raw Data** | Filterable table, CSV download, descriptive statistics |

---

## 🔢 FMCG KPIs Tracked

| KPI | Definition |
|-----|-----------|
| **Gross Income** | `total_price − (unit_price × quantity)` summed |
| **Customer Footfall** | Total transaction count (unique visits proxy) |
| **Average Basket Spend** | Mean `total_price` per transaction |
| **Gross Margin %** | `(Gross Income / Revenue) × 100` |
| **Member Conversion Rate** | % of transactions by loyalty Members |
| **Branch Performance Index** | Composite normalised score (revenue + footfall + margin) |
| **CLV Score** | Monetary-share-based lifetime value proxy per segment |

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Regressor (200 trees, max_depth=8)  
- **Target:** `total_price` (transaction revenue)  
- **Features:** `unit_price`, `quantity`, `tax`, `reward_points`, branch, city, customer_type, gender, product_category (label-encoded)  
- **Evaluation:** 80/20 train-test split + 5-fold cross-validation  
- **Comparison:** Random Forest vs Gradient Boosting vs Ridge Regression  

---

## 🔍 5 Key Findings

1. Member customers consistently outspend Normal customers by 15–30%.
2. The top product category accounts for >25% of gross income.
3. The highest-footfall city generates ~2× the revenue of the lowest.
4. Reward points show strong positive correlation with spend.
5. Random Forest achieves R² > 0.90 in sales prediction.

## ⚠️ 3 Risks

1. **Revenue Concentration** — Heavy dependence on 1–2 categories.
2. **Acquisition Imbalance** — Skewed Member/Normal ratio across branches.
3. **Thin Margins** — Low-margin categories vulnerable to cost inflation.

## 🚀 3 Opportunities

1. **Loyalty Upsell** — Tiered rewards (Silver/Gold/Platinum) to raise avg basket.
2. **Cross-City Expansion** — Micro-stores in low-footfall, high-margin cities.
3. **Category Mix Optimisation** — Introduce high-margin SKUs in underperforming branches.

## ✅ 5 Recommended Actions

| Timeframe | Action |
|-----------|--------|
| Immediate | Convert top-spending Normal customers to Member tier |
| 30-Day    | Dynamic pricing pilot for low-margin categories |
| 60-Day    | Deploy ML forecasting model for inventory replenishment |
| 90-Day    | Open 2 micro-stores in lowest-footfall cities |
| Ongoing   | Weekly KPI monitoring with alert thresholds |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML modelling |
| `python-docx` | Word report generation |
| `matplotlib` / `seaborn` | Static chart exports |

---

*Surbhi — Supermarket Sales Analytics Project*
