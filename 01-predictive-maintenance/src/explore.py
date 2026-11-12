import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Membaca dataset
df = pd.read_csv("data/ai4i2020.csv")

# 2. Melihat lima baris pertama
print("FIRST FIVE ROWS")
print(df.head())

# 3. Melihat ukuran dataset
print("\nDATASET SHAPE")
print(df.shape)

# 4. Melihat informasi dataset
print("\nDATASET INFORMATION")
df.info()

# 5. Melihat statistik deskriptif
print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

# 6. Memeriksa missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 7. Memeriksa data duplikat
print("\nDUPLICATE DATA")
print(df.duplicated().sum())

# 8. Distribusi target
print("\nMACHINE FAILURE DISTRIBUTION")
print(df["Machine failure"].value_counts())

# 9. Visualisasi target
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Observations")
plt.tight_layout()
plt.savefig("reports/failure_distribution.png")
plt.show()

# 10. analisi hubungan variabel
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
correlation = df[features + ["Machine failure"]].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Machine Variables")
plt.tight_layout()
plt.savefig("reports/correlation_matrix.png")
plt.show()
