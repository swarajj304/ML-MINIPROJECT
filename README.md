# ML-MINIPROJECT
# ❤️ Heart Disease Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Cleveland%20Heart%20Disease-lightgrey?logo=kaggle)](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

---

## 🧭 **Table of Contents**
1. [Overview](#overview)
2. [Objective](#objective)
3. [Methodology](#methodology)
4. [Results Summary](#results-summary)
5. [Installation & Setup](#installation--setup)
6. [Folder Structure](#folder-structure)
7. [Libraries Used](#libraries-used)
8. [Dataset](#dataset)
9. [References](#references)
10. [Conclusion](#conclusion)

---

## 🩺 **Overview**

This project reproduces the research paper  
**“Heart Disease Detection Using Machine Learning Models”** by *Amrit Singh et al.* (Elsevier, 2024).

The objective is to replicate and analyze the performance of multiple supervised machine learning models in predicting the likelihood of heart disease using the **Cleveland Heart Disease dataset** from Kaggle.

---

## 🎯 **Objective**

To reproduce the original paper’s methodology and evaluate several classical ML models  
(Logistic Regression, Decision Tree, Random Forest, SVM, kNN)  
for heart disease prediction using accuracy, precision, recall, F1-score, and ROC-AUC metrics.

---

## 🧩 **Methodology**

The project follows these major stages:

### 1️⃣ Data Collection
- **Dataset:** [Cleveland Heart Disease Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- Contains 303 samples and 14 medical attributes (age, sex, cholesterol, thalassemia, etc.).

### 2️⃣ Data Preprocessing
- Missing values replaced using median/mode imputation.  
- Duplicate rows removed and data standardized using `StandardScaler`.  
- Target variable converted to binary (0 = no disease, 1 = disease present).

### 3️⃣ Dataset Splitting
- Train-test split: **80% / 20%**
- Stratified sampling to preserve class balance.

### 4️⃣ Model Implementation
| Algorithm | Description |
|------------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based splits |
| Random Forest | Ensemble of decision trees |
| SVM (RBF Kernel) | Non-linear boundary learner |
| kNN | Distance-based classifier |

### 5️⃣ Hyperparameter Tuning
- Performed using `GridSearchCV` (5-fold cross-validation).  
- Parameter grid explored for:
  - `C`, `max_depth`, `n_estimators`, `n_neighbors`, `gamma`, etc.

### 6️⃣ Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Visualizations:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance (for tree-based models)

---

## 📊 **Results Summary**

| Model | Accuracy | ROC-AUC | Remarks |
|:------|:----------:|:--------:|:---------|
| Logistic Regression | 0.84 | 0.87 | Strong baseline |
| Decision Tree | 0.81 | 0.83 | Moderate accuracy |
| Random Forest | **0.90** | **0.92** | Best performance |
| SVM | 0.88 | 0.91 | Excellent generalization |
| kNN | 0.82 | 0.85 | Sensitive to scaling |

✅ **Best Model:** Random Forest Classifier  
All trained models are saved as `.pkl` files for reusability.

---

## ⚙️ **Installation & Setup**

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/heart-disease-detection.git
cd heart-disease-detection
