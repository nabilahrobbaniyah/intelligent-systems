
import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Industrial AI",
    layout="wide"
)

# Judul dashboard
st.title("Predictive Maintenance")
st.write(
    "AI-powered machine failure prediction "
    "using industrial sensor data."
)

# Memeriksa model
MODEL_PATH = "models/predictive_maintenance_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model not found. Please run src/train.py first."
    )
    st.stop()

# Memuat model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Sidebar
st.sidebar.header("Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0
)

rotation = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=5000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=150.0,
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=100
)

# Tombol prediksi
if st.sidebar.button("Predict Machine Condition"):

    # Membuat input DataFrame
    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rotation],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Prediksi
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    # Menampilkan hasil
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Failure Score",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if prediction == 1:
            st.error("Potential Failure Detected")
        else:
            st.success("No Failure Detected")

    # Informasi tambahan
    st.progress(float(probability))

    st.write(
        "This result is a machine learning prediction, "
        "not a confirmed mechanical diagnosis."
    )

    st.subheader("Input Parameters")
    st.dataframe(input_data)

    st.info(
        "Please use this prediction as a decision-support "
        "signal, not as a replacement for engineering inspection."
    )

else:
    st.info(
        "Enter machine parameters in the sidebar "
        "and click Predict Machine Condition."
    )
