
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# INDUSTRIAL ANOMALY DETECTION DASHBOARD
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    layout="wide"
)

st.title("Industrial Anomaly Detection")
st.write("AI-based machine sensor anomaly monitoring system.")

# 1. LOAD DATASET
st.sidebar.header("Dataset Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Industrial Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    default_path = "data/ai4i2020.csv"

    if not os.path.exists(default_path):
        st.warning("Please upload a CSV dataset.")
        st.stop()
    df = pd.read_csv(default_path)

# 2. SELECT FEATURES
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

missing = [col for col in features if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[features].copy()

# 3. DATA PREPROCESSING
if X.isnull().any().any():
    st.error("Dataset contains missing values. Please clean the data first.")
    st.stop()

# 4. MODEL CONFIGURATION
st.sidebar.header("Model Configuration")

contamination = st.sidebar.slider(
    "Expected Anomaly Percentage",
    min_value=0.01,
    max_value=0.15,
    value=0.03,
    step=0.01
)

# 5. TRAIN ISOLATION FOREST
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42
)

model.fit(X_scaled)
predictions = model.predict(X_scaled)

# Convert labels:
# 1 = Normal
# -1 = Anomaly
df["Anomaly"] = predictions

# 6. DASHBOARD METRICS
total = len(df)
anomalies = (df["Anomaly"] == -1).sum()
normal = (df["Anomaly"] == 1).sum()
anomaly_rate = anomalies / total * 100

st.subheader("Machine Monitoring Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Observations", f"{total:,}")
col2.metric("Normal Observations", f"{normal:,}")
col3.metric("Detected Anomalies", f"{anomalies:,}")

st.metric(
    "Anomaly Detection Rate",
    f"{anomaly_rate:.2f}%"
)

# 7. ANOMALY VISUALIZATION
st.subheader("Sensor Anomaly Visualization")
fig, ax = plt.subplots(figsize=(10, 5))
normal_data = df[df["Anomaly"] == 1]
anomaly_data = df[df["Anomaly"] == -1]

ax.scatter(
    normal_data["Rotational speed [rpm]"],
    normal_data["Torque [Nm]"],
    label="Normal",
    alpha=0.5
)

ax.scatter(
    anomaly_data["Rotational speed [rpm]"],
    anomaly_data["Torque [Nm]"],
    label="Anomaly",
    alpha=0.8,
    marker="x"
)

ax.set_xlabel("Rotational Speed (rpm)")
ax.set_ylabel("Torque (Nm)")
ax.set_title("Industrial Machine Anomaly Detection")
ax.legend()
st.pyplot(fig)

# 8. ANOMALY RESULTS
st.subheader("Detection Results")
result = df[df["Anomaly"] == -1]
st.dataframe(
    result[features + ["Anomaly"]],
    use_container_width=True
)

# 9. DOWNLOAD RESULTS
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Anomaly Results",
    data=csv,
    file_name="industrial_anomaly_results.csv",
    mime="text/csv"
)