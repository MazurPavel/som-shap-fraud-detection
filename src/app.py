import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go

from minisom import MiniSom
import xgboost as xgb


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="SOM + SHAP Fraud Detection Demo",
    page_icon="🔍",
    layout="wide"
)


# ---------------------------
# Load models & data
# ---------------------------
@st.cache_resource
def load_models():
    with open("models/som_model.pkl", "rb") as f:
        som_model = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/shap_explainer.pkl", "rb") as f:
        shap_explainer = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return som_model, xgb_model, shap_explainer, scaler, feature_cols


@st.cache_data
def load_data():
    return pd.read_csv("data/demo_vesta.csv")


som_model, xgb_model, shap_explainer, scaler, feature_cols = load_models()
df = load_data()


# ---------------------------
# Title & intro
# ---------------------------
st.title("🔍 SOM + SHAP Fraud Detection – Live Demo (Vesta)")

st.markdown("""
This demo illustrates the **core idea** of my thesis:

- SOM clusters transactions into behavioral patterns  
- A surrogate XGBoost model predicts fraud risk  
- SHAP explains *why* a specific transaction looks suspicious  
""")


# ---------------------------
# Transaction selection
# ---------------------------
st.sidebar.header("Transaction selection")

max_rows = min(100, len(df))
row_index = st.sidebar.number_input(
    "Choose transaction index:",
    min_value=0,
    max_value=max_rows - 1,
    value=0,
    step=1
)

transaction = df.iloc[row_index]


# ---------------------------
# Prepare features
# ---------------------------
features = transaction[feature_cols]
X = features.values.reshape(1, -1)


# ---------------------------
# SOM + XGBoost
# ---------------------------
bmu = som_model.winner(X[0])
bmu_x, bmu_y = bmu

pred = xgb_model.predict(X)[0]

if hasattr(xgb_model, "predict_proba"):
    proba = xgb_model.predict_proba(X)[0]
    risk_score = proba[1] if len(proba) > 1 else proba[0]
else:
    risk_score = float(pred)


# ---------------------------
# SHAP values
# ---------------------------
num_features = X.shape[1]
min_evals = 2 * num_features + 1
max_evals = min_evals + 50

shap_explanation = shap_explainer(
    X,
    max_evals=max_evals,
    main_effects=False
)

shap_vals_sample = shap_explanation.values[0]


# ---------------------------
# TWO-COLUMN LAYOUT
# ---------------------------

st.markdown("---")
colA, colB = st.columns([1.3, 1])

with colA:
    st.write("### 🔎 Selected transaction")
    st.dataframe(transaction.to_frame().T)

    st.write("### Model output")
    st.metric("SOM BMU", f"({bmu_x}, {bmu_y})")

    status = "⚠️ HIGH RISK" if pred == 1 else "✅ NORMAL"
    st.metric("Prediction", status)

    st.metric("Risk score", f"{risk_score:.2%}")


with colB:
    st.write("### 📊 SHAP: Top contributing features (interactive)")

    # --- Подготовка данных для топ-10 фич ---
    raw_vals = shap_vals_sample
    abs_vals = np.abs(raw_vals)
    feature_names = features.index.tolist()

    sorted_idx = np.argsort(abs_vals)[-10:][::-1]

    top_features = [feature_names[i] for i in sorted_idx]
    top_shap_raw = [raw_vals[i] for i in sorted_idx]
    top_shap_abs = [abs_vals[i] for i in sorted_idx]

    # Цвета: красный = повышает риск, синий = снижает риск
    colors = [
        "crimson" if v > 0 else "royalblue"
        for v in top_shap_raw
    ]

    # --- Двухцветный горизонтальный график ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_shap_abs[::-1],             # длина бара = |SHAP|
        y=top_features[::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "SHAP value: %{customdata[0]:.4f}<br>" +
            "|SHAP| impact: %{x:.4f}<extra></extra>"
        ),
        customdata=[[v] for v in top_shap_raw[::-1]]
    ))

    fig.update_layout(
        height=380,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="|SHAP| impact",
        yaxis_title="Feature",
        title="Top-10 most influential features"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Red bars indicate features that **increase** the fraud risk, "
        "while blue bars indicate features that **decrease** it."
    )

    # --- Интерактивная таблица факторов риска ---
    st.write("### 🔬 Detailed feature impact table")

    directions = [
        "↑ increases risk" if v > 0 else "↓ decreases risk"
        for v in top_shap_raw
    ]

    table_df = pd.DataFrame({
        "Feature": top_features,
        "SHAP value": np.round(top_shap_raw, 6),
        "|SHAP| impact": np.round(top_shap_abs, 6),
        "Direction": directions
    })

    st.dataframe(
        table_df,
        use_container_width=True
    )

    st.caption(
        "This table summarises the top behavioural signals driving the model's decision "
        "for the selected transaction."
    )
