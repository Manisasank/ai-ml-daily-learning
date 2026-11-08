"""
Model Evaluation and Inference Module for Customer Churn Prediction
Computes comprehensive classification metrics and provides real-time customer risk scoring.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from .preprocess import prepare_data
    from .model import train_models
except ImportError:
    from preprocess import prepare_data
    from model import train_models


def evaluate_predictions(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Calculate full set of evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def predict_customer_risk(customer_profile: dict, model, scaler, feature_names) -> Dict[str, Any]:
    """
    Predict churn risk for an individual customer profile.
    """
    df = pd.DataFrame([customer_profile])
    
    # Feature engineering
    safe_denom = df["tenure_months"].iloc[0] * df["monthly_charges"].iloc[0]
    df["charge_ratio"] = df["total_charges"].iloc[0] / (safe_denom if safe_denom > 0 else 1.0)
    df["is_long_term"] = int(df["tenure_months"].iloc[0] >= 24)
    df["has_security_bundle"] = int(
        (df["tech_support"].iloc[0] == "Yes") and (df["online_security"].iloc[0] == "Yes")
    )
    
    # One-hot encoding
    encoded = pd.get_dummies(df, columns=["contract", "payment_method", "tech_support", "online_security"])
    
    # Align columns
    for col in feature_names:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_names]
    
    # Scale continuous
    continuous_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "charge_ratio"]
    encoded[continuous_cols] = scaler.transform(encoded[continuous_cols])
    
    prob = model.predict_proba(encoded)[0, 1]
    is_churn = int(prob >= 0.5)
    
    risk_level = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.35 else "Low")
    
    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": is_churn,
        "risk_level": risk_level
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scaler = prepare_data()
    models = train_models(X_train, y_train)
    best_model = models["LogisticRegression"]["model"]
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Test Set Evaluation Report ---")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    
    # Sample customer risk assessment
    sample_customer = {
        "age": 45,
        "tenure_months": 2,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "tech_support": "No",
        "online_security": "No",
        "monthly_charges": 98.50,
        "total_charges": 197.00
    }
    risk = predict_customer_risk(sample_customer, best_model, scaler, features)
    print("\n--- Sample Customer Risk Score ---")
    print(f"Profile: Month-to-month, High Monthly Charges ($98.50), No Support")
    print(f"Result: {risk}")
