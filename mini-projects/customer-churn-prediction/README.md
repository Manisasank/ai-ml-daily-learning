# End-to-End Customer Churn Prediction System

A complete machine learning project predicting customer churn probability and assigning actionable retention risk tiers.

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a company. Acquiring new customers typically costs 5–7x more than retaining existing ones. This project demonstrates a production-style machine learning pipeline to:
1. Generate realistic customer subscription and billing data with probabilistic risk profiles.
2. Engineer informative behavioral indicators (billing ratios, contract tiers, support bundles).
3. Train and cross-validate supervised classifiers (`LogisticRegression` and `RandomForest`).
4. Evaluate models with comprehensive metrics (Accuracy, ROC-AUC, Precision-Recall, Confusion Matrix).
5. Expose real-time single-customer risk scoring with interpretability tiers (Low, Moderate, High).

---

## 🏗️ Architecture

```
mini-projects/customer-churn-prediction/
├── data_generator.py   # Synthetic demographic and billing data generator
├── preprocess.py       # Feature engineering, encoding, and scaling pipeline
├── model.py            # Model training, 5-fold cross validation, and selection
├── evaluate.py         # Evaluation metrics, confusion matrix, and inference CLI
└── README.md           # Project documentation and benchmarks
```

---

## ⚙️ Features & Data Dictionary

| Feature Name | Description | Transformation |
| :--- | :--- | :--- |
| `age` | Customer age in years | StandardScaler |
| `tenure_months` | Months subscribed with service | StandardScaler |
| `monthly_charges` | Current monthly plan rate in USD | StandardScaler |
| `total_charges` | Cumulative billings to date | StandardScaler |
| `charge_ratio` | Ratio of actual billings to expected tenure baseline | Engineered / Standardized |
| `is_long_term` | Indicator for tenure >= 24 months | Binary indicator |
| `has_security_bundle` | Both Tech Support and Online Security active | Binary indicator |
| `contract` | Month-to-month, One year, Two year | One-Hot Encoded |
| `payment_method` | Electronic check, Mailed check, Credit card, Bank transfer | One-Hot Encoded |

---

## 📊 Benchmark Results

| Model | 5-Fold ROC-AUC | Test Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** (L2) | **0.7468** | **77.0%** | **0.56** |
| **Random Forest** (Depth=6) | 0.7256 | 76.4% | 0.54 |

### Risk Factor Insights:
- **Highest Churn Drivers**: Month-to-month contract type, Electronic check payment, High monthly fees (> $70/mo).
- **Strongest Retention Anchors**: Long tenure (> 24 months), Two-year contract commitments, Active Tech Support & Online Security add-ons.

---

## 🚀 Quickstart & Usage

### 1. Generate Synthetic Data
```bash
python data_generator.py
```

### 2. Run Preprocessing Pipeline
```bash
python preprocess.py
```

### 3. Train & Select Best Model
```bash
python model.py
```

### 4. Evaluate & Run Inference
```bash
python evaluate.py
```

---

## 🔮 Future Improvements
- [ ] Add XGBoost and LightGBM gradient boosting algorithms.
- [ ] Implement SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
- [ ] Incorporate SHAP (SHapley Additive exPlanations) for local explainability plots.
