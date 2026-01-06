import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import streamlit as st
import numpy as np
import pandas as pd
import joblib

from src.strategy.degradation_model import build_degradation_models

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="F1 Strategy Simulator",
    layout="wide"
)

st.title("🏎️ F1 Strategy Simulator")
st.markdown(
    "Interactive strategy tool combining **ML race outcome prediction** "
    "and **tire degradation modeling**."
)

# --------------------------------------------------
# Load models (cached)
# --------------------------------------------------
@st.cache_resource
def load_podium_model():
    return joblib.load("models/podium_model.pkl")

@st.cache_resource
def load_degradation_models():
    return build_degradation_models()

podium_model = load_podium_model()
degradation_models = load_degradation_models()

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Race Context")

grid_position = st.sidebar.slider(
    "Starting Grid Position",
    min_value=1,
    max_value=20,
    value=5
)

compound = st.sidebar.selectbox(
    "Tire Compound",
    options=list(degradation_models.keys())
)

max_laps = st.sidebar.slider(
    "Stint Length (laps)",
    min_value=5,
    max_value=40,
    value=20
)

noise = st.sidebar.checkbox(
    "Add race noise",
    value=False
)

# --------------------------------------------------
# Phase 3: Degradation model → lap time prediction
# --------------------------------------------------
model_info = degradation_models[compound]
degradation_model = model_info["model"]

laps = np.arange(1, max_laps + 1).reshape(-1, 1)
lap_times = degradation_model.predict(laps)

if noise:
    lap_times = lap_times + np.random.normal(0, 0.15, size=len(lap_times))

# --------------------------------------------------
# Phase 2: Feature engineering for ML model
# --------------------------------------------------
features = pd.DataFrame([{
    "grid_position": grid_position,
    "avg_lap_time": float(lap_times.mean()),
    "best_lap_time": float(lap_times.min()),
    "lap_count": max_laps,
    "stints_used": 1,
    "used_soft": int(compound == "SOFT"),
    "used_medium": int(compound == "MEDIUM"),
    "used_hard": int(compound == "HARD"),
}])

# --------------------------------------------------
# Phase 2: Podium probability prediction
# --------------------------------------------------
podium_prob = podium_model.predict_proba(features)[0, 1]

# --------------------------------------------------
# UI: Podium probability metric
# --------------------------------------------------
st.subheader("🏆 Race Outcome Prediction")

st.metric(
    label="Podium Probability",
    value=f"{podium_prob * 100:.1f}%"
)

st.markdown(
    f"""
    **Interpretation:**  
    Starting from **P{grid_position}** on **{compound}** tires for a  
    **{max_laps}-lap stint** gives an estimated  
    **{podium_prob * 100:.1f}% chance of finishing on the podium**.
    """
)

# --------------------------------------------------
# UI: Degradation curve visualization
# --------------------------------------------------
st.subheader(f"📈 Tire Degradation Curve — {compound}")

df_plot = pd.DataFrame({
    "Lap": laps.flatten(),
    "Predicted Lap Time (s)": lap_times
})

st.line_chart(
    df_plot.set_index("Lap"),
    use_container_width=True
)

# --------------------------------------------------
# UI: Compound comparison table
# --------------------------------------------------
st.subheader("🔍 Compound Comparison")

comparison_rows = []

for comp, info in degradation_models.items():
    m = info["model"]
    time_end = m.predict([[max_laps]])[0]

    comparison_rows.append({
        "Compound": comp,
        f"Lap Time @ Lap {max_laps} (s)": round(time_end, 2),
        "Degradation Rate (s/lap)": round(info["slope"], 4),
        "Training Samples": info["n_samples"]
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True)

# --------------------------------------------------
# Footer explanation
# --------------------------------------------------
st.markdown(
    """
    ### 🧠 How to use this tool
    - **Grid position** strongly influences podium probability  
    - **SOFT tires** give pace but degrade faster  
    - **HARD tires** are consistent but slower  
    - The model combines **ML predictions** with **tire degradation physics**

    This mirrors real-world Formula 1 strategy decision making.
    """
)
