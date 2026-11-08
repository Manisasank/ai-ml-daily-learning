"""
Customer Churn Preprocessing & Feature Engineering Pipeline
Handles data ingestion, feature transformation, categorical encoding,
and standard scaling for machine learning modeling.
"""

import os
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .data_generator import generate_churn_dataset
except ImportError:
    from data_generator import generate_churn_dataset


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-relevant features from raw customer profiles.
    
    New Features:
        - avg_monthly_ratio: total_charges / (tenure_months * monthly_charges)
        - is_long_term: tenure_months > 24
        - has_security_bundle: both tech support and online security active
    """
    df = df.copy()
    
    # Avoid division by zero
    safe_denominator = np_safe = df["tenure_months"] * df["monthly_charges"]
    safe_denominator = safe_denominator.replace(0, 1.0)
    df["charge_ratio"] = df["total_charges"] / safe_denominator
    
    df["is_long_term"] = (df["tenure_months"] >= 24).astype(int)
    df["has_security_bundle"] = ((df["tech_support"] == "Yes") & (df["online_security"] == "Yes")).astype(int)
    
    return df


def prepare_data(
    csv_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], StandardScaler]:
    """
    Full pipeline to clean, encode, and split dataset into train and test sets.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_churn_dataset(n_samples=2500, random_state=random_state)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Drop identifier
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
        
    y = df["churn"]
    X = df.drop(columns=["churn"])
    
    # One-hot encode categorical variables
    categorical_cols = ["contract", "payment_method", "tech_support", "online_security"]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = list(X_encoded.columns)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize continuous variables
    continuous_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "charge_ratio"]
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, _ = prepare_data()
    print(f"Dataset successfully processed.")
    print(f"Training features shape: {X_tr.shape}")
    print(f"Testing features shape: {X_te.shape}")
    print(f"Features list ({len(feats)}): {feats}")
