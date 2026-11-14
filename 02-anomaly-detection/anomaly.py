import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

os.makedirs("reports", exist_ok=True)

# 1. Membaca dataset
df = pd.read_csv("data/ai4i2020.csv")

# 2. Memilih fitur sensor
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features].copy()

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Membuat model
model = IsolationForest(
    n_estimators=100,
    contamination=0.03,
    random_state=42
)

# 5. Melatih model
model.fit(X_scaled)

# 6. Mendeteksi anomali
predictions = model.predict(X_scaled)

# Isolation Forest:
# 1 = Normal
# -1 = Anomaly
df["Anomaly"] = predictions

# 7. Menghitung hasil
print(df["Anomaly"].value_counts())

# 8. Visualisasi
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Rotational speed [rpm]"],
    df["Torque [Nm]"],
    c=df["Anomaly"],
    alpha=0.6
)

plt.xlabel("Rotational Speed (rpm)")
plt.ylabel("Torque (Nm)")
plt.title("Industrial Machine Anomaly Detection")

plt.savefig("reports/anomaly_detection.png")
plt.show()

# 9. Menyimpan hasil
df.to_csv("reports/anomaly_results.csv", index=False)
