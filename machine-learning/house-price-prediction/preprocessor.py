"""
Housing Feature Engineering & Preprocessing Pipeline
Transforms raw structural/location attributes into scaled regression features
and handles target log-transformation.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_generator import generate_housing_data
except ImportError:
    from dataset_generator import generate_housing_data


def engineer_housing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain indicators from structural attributes.
    """
    df = df.copy()
    
    # 1. Room density and space ratios
    total_rooms = df["bedrooms"] + df["bathrooms"]
    df["sqft_per_room"] = df["sqft_living"] / np.maximum(total_rooms, 1.0)
    df["lot_living_ratio"] = df["lot_size_sqft"] / np.maximum(df["sqft_living"], 1.0)
    
    # 2. Amenities composite
    df["amenities_count"] = df["has_garage"] + df["has_pool"]
    
    # 3. Non-linear age penalty (accelerated depreciation initially, leveling off)
    df["age_squared"] = (df["house_age_years"] / 10.0) ** 2
    
    # 4. Location-School interaction
    df["school_dist_interaction"] = df["school_rating"] / np.maximum(df["dist_city_miles"], 1.0)
    
    return df


def prepare_regression_data(
    n_samples: int = 2500,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Full pipeline to generate, engineer features, scale, and split housing data.
    Target values are log-transformed (np.log1p) to normalize skewed pricing distributions.
    """
    raw_df = generate_housing_data(n_samples=n_samples, random_state=random_state)
    engineered_df = engineer_housing_features(raw_df)
    
    # Separate features and target
    y_raw = engineered_df["price"].values
    y_log = np.log1p(y_raw)
    X_raw = engineered_df.drop(columns=["price"])
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_raw, columns=["neighborhood"], drop_first=True)
    feature_names = list(X_encoded.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_log, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    continuous_features = [
        "sqft_living", "bedrooms", "bathrooms", "lot_size_sqft",
        "house_age_years", "school_rating", "dist_city_miles",
        "sqft_per_room", "lot_living_ratio", "age_squared", "school_dist_interaction"
    ]
    
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, sc = prepare_regression_data()
    print("Preprocessing completed.")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Features ({len(feats)}): {feats}")
    print(f"Mean target (log): {np.mean(y_tr):.4f} -> Approx Price: ${np.expm1(np.mean(y_tr)):,.2f}")
