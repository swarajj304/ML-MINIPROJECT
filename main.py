"""
Heart Disease Detection using Machine Learning
Reproduction of "Heart Disease Detection Using Machine Learning Models" (Elsevier, 2024)
Group 80 | Batch A3 | Roll No. 16014223086
Institution: K. J. Somaiya Institute of Engineering, Somaiya Vidyavihar University
"""

# ==============================================================
# 1. Import Libraries
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ==============================================================
# 2. Load and Preprocess Dataset
# ==============================================================

print("\n📥 Loading dataset...")
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv("processed_cleveland.csv", names=columns)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Handle duplicates and convert to numeric
df.drop_duplicates(inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
for c in df.select_dtypes(include='number').columns:
    df[c].fillna(df[c].median(), inplace=True)

# Create binary target variable
df["target"] = (df["num"] > 0).astype(int)

print("✅ Preprocessing completed successfully.")

# ==============================================================
# 3. Split Dataset
# ==============================================================

X = df.drop(columns=["target", "num"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ==============================================================
# 4. Define Models and Hyperparameter Grids
# ==============================================================

models = {
    "Logistic Regression": (
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
        {"C": [0.1, 1, 10]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9]}
    )
}

# ==============================================================
# 5. Train and Evaluate Models
# ==============================================================

results = []

print("\n🚀 Training models...")
for name, (model, params) in models.items():
    print(f"\n▶ Training {name}...")
    grid = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    # Evaluation metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    results.append(metrics)

    # Save model
    joblib.dump(best_model, f"{name.replace(' ', '_')}_model.pkl")
    print(f"✅ {name} completed | Accuracy: {metrics['Accuracy']:.3f}, AUC: {metrics['ROC_AUC']:.3f}")

# ==============================================================
# 6. Results Summary
# ==============================================================

results_df = pd.DataFrame(results)
print("\n📊 Model Performance Summary:")
print(results_df)

results_df.to_csv("results_summary.csv", index=False)
print("📁 Results saved as results_summary.csv")

# ==============================================================
# 7. Visualizations
# ==============================================================

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Comparison - Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="ROC_AUC", data=results_df)
plt.title("Model Comparison - ROC-AUC")
plt.tight_layout()
plt.show()

# Confusion Matrix and ROC Curve for Best Model
best_model_name = results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")

best_model = joblib.load(f"{best_model_name.replace(' ', '_')}_model.pkl")

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

try:
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC Curve - {best_model_name}")
    plt.show()
except Exception as e:
    print("ROC curve not available:", e)

print("\n Execution completed successfully.")
