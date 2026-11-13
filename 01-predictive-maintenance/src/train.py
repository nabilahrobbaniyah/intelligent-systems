
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Membaca dataset
df = pd.read_csv("data/ai4i2020.csv")

# 2. Menentukan fitur
numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

categorical_features = ["Type"]

features = numeric_features + categorical_features

# 3. Memisahkan fitur dan target
X = df[features]
y = df["Machine failure"]

# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# 6. Membuat model Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 7. Membuat pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# 8. Melatih model
pipeline.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")

# 9. Prediksi
y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]

# 10. Evaluasi
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, y_probability),
    "pr_auc": average_precision_score(
        y_test, y_probability
    )
}

print("\nMODEL EVALUATION")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# 11. Menyimpan model
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    pipeline,
    "models/predictive_maintenance_model.pkl"
)

# 12. Menyimpan hasil evaluasi
with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMODEL SAVED SUCCESSFULLY")
