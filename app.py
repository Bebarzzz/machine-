import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Suppress TF info logs before importing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from catboost import CatBoostClassifier
from tensorflow import keras

# Make wrapper.py importable so the pickled Stacking model can be unpickled
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrapper import KerasClassifierWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chronotype Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px; }
      div[data-testid="stMetric"] {
          background: #f7f9fc;
          padding: 16px 18px;
          border-radius: 12px;
          border: 1px solid #e6eaf0;
      }
      div[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #5b6573; font-weight: 600; }
      div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #1f2937; }
      div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
      .acc-line { font-size: 0.78rem; color: #6b7280; margin-top: 6px; }
      .acc-line b { color: #1f2937; }
      section[data-testid="stSidebar"] h2 { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load artifacts (cached for the session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    rf      = joblib.load("saved_models/rf_model.pkl")
    xgb     = joblib.load("saved_models/xgb_model.pkl")
    svm     = joblib.load("saved_models/svm_model.pkl")
    scaler  = joblib.load("saved_models/scaler.pkl")
    le      = joblib.load("saved_models/label_encoder.pkl")
    info    = joblib.load("saved_models/feature_info.pkl")

    cb = CatBoostClassifier()
    cb.load_model("saved_models/catboost_model.cbm")

    nn = keras.models.load_model("saved_models/nn_model.keras")

    stacking = None
    if info.get("stacking_saved", False) and os.path.exists("saved_models/stacking_model.pkl"):
        stacking = joblib.load("saved_models/stacking_model.pkl")
        # Re-attach the loaded Keras model inside the wrapper(s)
        for est in stacking.estimators_:
            if isinstance(est, KerasClassifierWrapper) and est.model is None:
                est.model = nn

    return {
        "rf": rf, "xgb": xgb, "svm": svm, "nn": nn, "cb": cb,
        "stacking": stacking, "scaler": scaler, "le": le, "info": info,
    }


try:
    art = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Saved models not found ({e}). Open **project.ipynb** and run the "
        "**Save Models for GUI** cell first, then refresh this page."
    )
    st.stop()

INFO = art["info"]
FEATURES = INFO["feature_names"]
STATS = INFO["stats"]
ACC = INFO["accuracies"]
SCALED_MODELS = set(INFO.get("scaled_models", []))
LE = art["le"]
SCALER = art["scaler"]

CHRONO_LABELS = {
    1.0: "Strong Morning Type",
    2.0: "Morning Type",
    3.0: "Intermediate",
    4.0: "Evening Type",
    5.0: "Irregular / Shift Pattern",
}

FEATURE_META = {
    "Age":       ("Age", "years"),
    "BMI":       ("Body Mass Index", "kg/m²"),
    "Waist_C":   ("Waist Circumference", "cm"),
    "Systolic":  ("Systolic Blood Pressure", "mmHg"),
    "Diastolic": ("Diastolic Blood Pressure", "mmHg"),
    "Carb_diet": ("Daily Carbohydrate Intake", "g"),
    "HSCRP":     ("hs-CRP (Inflammation)", "mg/L"),
    "Alcohol":   ("Alcohol Intake", "drinks/day"),
    "Sleep_hrs": ("Sleep Duration", "hours"),
    "WakeUpCat": ("Wake-up Category", ""),
}

# Wake-up category labels (derived from training-data wake-time distribution)
WAKEUP_CAT_LABELS = {
    1: "1 — Very early (≤ 5 AM)",
    2: "2 — Early morning (5:00 – 6:30 AM)",
    3: "3 — Normal morning (6:30 – 8:30 AM)",
    4: "4 — Late morning (8:45 – 9:30 AM)",
    5: "5 — Very late (10:30 AM or later)",
    9: "9 — Unknown / missing",
}

# Order matters: defines card layout (2 rows x 3 cols)
MODEL_ORDER = [
    "Random Forest", "XGBoost", "CatBoost",
    "SVM", "Neural Network", "Stacking",
]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌙 Chronotype Predictor")
st.caption(
    "Predict an individual's sleep chronotype from health and lifestyle metrics. "
    "Six models trained on NHANES data are loaded below."
)

with st.expander("ℹ️  Why does chronotype matter?  (click to expand)"):
    st.markdown(
        """
        **Chronotype** is a person's natural inclination toward morning or evening
        activity — when they prefer to sleep, wake, eat, and perform mentally.
        Knowing someone's chronotype is more than a curiosity; it carries
        well-documented implications across health, behavior, and clinical care.

        **What we can infer from `Chronotype_slphrs`:**

        | Domain | What chronotype tells us |
        |---|---|
        | **Sleep behavior** | Habitual bedtime, wake time, and total sleep duration |
        | **Metabolic health** | Evening types show **higher BMI, waist circumference, blood pressure, and inflammation (hs-CRP)**; greater risk of type 2 diabetes |
        | **Cardiovascular risk** | Late chronotypes have elevated risk of hypertension and cardiovascular events |
        | **Mental health** | Evening types have **2–3× higher rates** of depression, anxiety, and substance use disorders |
        | **Diet & lifestyle** | Later chronotypes consume more late-day carbohydrates and alcohol |
        | **Cognitive performance** | Peak alertness differs by hours (morning types ~10 AM, evening ~6 PM) |
        | **Occupational fit** | Irregular pattern (type 5) often indicates shift work — itself a WHO-classified risk factor |

        **Why predict it from routine health data?**

        Chronotype is normally measured with sleep questionnaires that clinics
        rarely collect. If we can estimate it from data already in a patient's
        chart — BMI, blood pressure, inflammation markers, diet, sleep hours —
        chronotype becomes a **passive, no-cost screening signal** for
        metabolic and cardiovascular risk.

        ---

        **Class meanings** (derived from the NHANES dataset):

        - **Type 1 — Strong Morning** &nbsp;·&nbsp; bed ~21:00, wake ~05:30, 8.5 h sleep
        - **Type 2 — Morning** &nbsp;·&nbsp; bed ~22:00, wake ~06:00, 8.0 h sleep
        - **Type 3 — Intermediate** &nbsp;·&nbsp; bed ~23:00, wake ~07:00, 7.5 h sleep (most common)
        - **Type 4 — Evening** &nbsp;·&nbsp; bed ~01:00, wake ~08:00, 6.5 h sleep
        - **Type 5 — Irregular / Shift** &nbsp;·&nbsp; bed ~18:45, wake ~07:00 — unusual schedule, often shift workers or fragmented sleepers
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Patient Inputs")
st.sidebar.caption("Sliders are bounded by the training-data range.")

inputs = {}
for feat in FEATURES:
    label, unit = FEATURE_META.get(feat, (feat, ""))
    s = STATS[feat]
    display = f"{label} ({unit})" if unit else label

    if feat == "WakeUpCat":
        inputs[feat] = st.sidebar.selectbox(
            display,
            options=[1, 2, 3, 4, 5, 9],
            index=2,  # default: category 3 (typical morning waker)
            format_func=lambda x: WAKEUP_CAT_LABELS[x],
        )
    else:
        # Features that benefit from decimal precision in the GUI
        float_features = {"BMI", "Waist_C", "Systolic", "Diastolic", "HSCRP"}
        if feat in float_features or (s["max"] - s["min"]) < 10:
            step, fmt = 0.1, "%.1f"
        else:
            step, fmt = 1.0, "%.0f"
        inputs[feat] = st.sidebar.slider(
            display,
            min_value=float(s["min"]),
            max_value=float(s["max"]),
            value=float(s["median"]),
            step=step,
            format=fmt,
        )

predict_btn = st.sidebar.button(
    "Predict Chronotype", type="primary", use_container_width=True
)

st.sidebar.divider()
st.sidebar.subheader("Model accuracies")
acc_table = pd.DataFrame(
    {
        "Model": MODEL_ORDER,
        "Train": [f"{ACC[m]['train']*100:.1f}%" for m in MODEL_ORDER],
        "Test":  [f"{ACC[m]['test']*100:.1f}%"  for m in MODEL_ORDER],
    }
)
st.sidebar.dataframe(acc_table, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Build input matrix (raw + scaled versions)
# ---------------------------------------------------------------------------
X_raw = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
X_scaled = SCALER.transform(X_raw)

if not predict_btn:
    st.info(
        "Adjust the patient inputs in the sidebar, then click "
        "**Predict Chronotype** to see the model predictions."
    )
    with st.expander("Current input values"):
        st.dataframe(X_raw.T.rename(columns={0: "Value"}), use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Run predictions for each model
# ---------------------------------------------------------------------------
def to_chrono(label_value: float) -> str:
    return CHRONO_LABELS.get(float(label_value), f"Type {label_value:g}")


def decode(idx) -> float:
    """Convert encoded label idx (0–4) back to chronotype value (1.0–5.0)."""
    return float(LE.inverse_transform([int(idx)])[0])


predictions = {}  # model_name -> (predicted_chronotype, proba_array, confidence)

# Random Forest — trained on raw features, predicts 1.0–5.0 directly
rf_proba = art["rf"].predict_proba(X_raw)[0]
predictions["Random Forest"] = (float(art["rf"].predict(X_raw)[0]), rf_proba)

# XGBoost — raw features, predicts 0–4
xgb_idx = int(art["xgb"].predict(X_raw)[0])
predictions["XGBoost"] = (decode(xgb_idx), art["xgb"].predict_proba(X_raw)[0])

# CatBoost — raw features, predicts 0–4
cb_idx = int(np.asarray(art["cb"].predict(X_raw)).flatten()[0])
predictions["CatBoost"] = (decode(cb_idx), art["cb"].predict_proba(X_raw)[0])

# SVM — scaled features
svm_idx = int(art["svm"].predict(X_scaled)[0])
predictions["SVM"] = (decode(svm_idx), art["svm"].predict_proba(X_scaled)[0])

# Neural Network — scaled features, softmax outputs
nn_proba = art["nn"].predict(X_scaled, verbose=0)[0]
predictions["Neural Network"] = (decode(int(np.argmax(nn_proba))), nn_proba)

# Stacking — scaled features (may be unavailable if save failed)
if art["stacking"] is not None:
    st_idx = int(art["stacking"].predict(X_scaled)[0])
    predictions["Stacking"] = (decode(st_idx), art["stacking"].predict_proba(X_scaled)[0])
else:
    predictions["Stacking"] = None

# ---------------------------------------------------------------------------
# Display — 2 rows x 3 cols of model cards
# ---------------------------------------------------------------------------
st.subheader("Predictions")

row1 = st.columns(3)
row2 = st.columns(3)
slots = row1 + row2

for slot, name in zip(slots, MODEL_ORDER):
    with slot:
        result = predictions.get(name)
        if result is None:
            st.metric(label=name, value="Unavailable", delta="model not loaded", delta_color="off")
            continue

        pred, proba = result
        confidence = float(np.max(proba))
        st.metric(
            label=name,
            value=to_chrono(pred),
            delta=f"{confidence*100:.1f}% confidence",
            delta_color="off",
        )
        st.markdown(
            f"<div class='acc-line'>Train <b>{ACC[name]['train']*100:.1f}%</b> "
            f"&nbsp;·&nbsp; Test <b>{ACC[name]['test']*100:.1f}%</b></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Probability comparison across all models
# ---------------------------------------------------------------------------
st.subheader("Class probabilities — all models")

class_labels = [CHRONO_LABELS.get(float(c), f"Type {c:g}") for c in LE.classes_]
proba_columns = {}
for name in MODEL_ORDER:
    result = predictions.get(name)
    if result is not None:
        proba_columns[name] = result[1]

proba_df = pd.DataFrame(proba_columns, index=class_labels)
proba_df.index.name = "Chronotype"

st.bar_chart(proba_df, height=340)

with st.expander("Raw probability table"):
    st.dataframe(proba_df.style.format("{:.3f}"), use_container_width=True)
