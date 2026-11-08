"""
Customer Churn Model Training & Selection
Trains Logistic Regression and Random Forest models with cross-validation
and feature importance extraction.
"""

import os
import pickle
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

try:
    from .preprocess import prepare_data
except ImportError:
    from preprocess import prepare_data


def train_models(X_train, y_train) -> Dict[str, Any]:
    """
    Train and compare multiple models with 5-fold cross-validation.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std())
        }
        print(f"[{name}] 5-Fold ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
    return results


def save_best_model(model, scaler, feature_names, file_path: str = "churn_model_bundle.pkl"):
    """
    Persist best performing model and preprocessing metadata.
    """
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    with open(file_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Best model bundle saved to: {file_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scaler = prepare_data()
    trained = train_models(X_train, y_train)
    
    # Pick the model with highest CV score
    best_name = max(trained.keys(), key=lambda k: trained[k]["cv_roc_auc_mean"])
    best_model = trained[best_name]["model"]
    print(f"\nChampion Model: {best_name}")
    
    # Feature importance for Random Forest
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop 5 Feature Importances:")
        for i in range(min(5, len(features))):
            idx = indices[i]
            print(f"  {features[idx]}: {importances[idx]:.4f}")
            
    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(out_dir, "churn_model_bundle.pkl")
    save_best_model(best_model, scaler, features, save_path)
