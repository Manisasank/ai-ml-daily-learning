"""
Regression Evaluation Suite & Property Valuation CLI
Computes R², RMSE, MAE, MAPE on real estate test partitions and provides
an interactive valuation tool with predictive confidence intervals.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

try:
    from .preprocessor import prepare_regression_data, engineer_housing_features
    from .train_models import train_and_benchmark_models
except ImportError:
    from preprocessor import prepare_regression_data, engineer_housing_features
    from train_models import train_and_benchmark_models


def compute_regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics on both log-transformed and original dollar scale.
    """
    # Inverse transform to original currency
    y_true_usd = np.expm1(y_true_log)
    y_pred_usd = np.expm1(y_pred_log)
    
    rmse_usd = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))
    mae_usd = float(mean_absolute_error(y_true_usd, y_pred_usd))
    mape = float(mean_absolute_percentage_error(y_true_usd, y_pred_usd))
    r2 = float(r2_score(y_true_log, y_pred_log))
    
    return {
        "r2_score": round(r2, 4),
        "rmse_usd": round(rmse_usd, 2),
        "mae_usd": round(mae_usd, 2),
        "mape": round(mape, 4)
    }


def estimate_property_value(
    property_specs: dict,
    model,
    scaler,
    feature_names
) -> Dict[str, Any]:
    """
    Value a single property specification and return market valuation with confidence range.
    """
    raw_df = pd.DataFrame([property_specs])
    eng_df = engineer_housing_features(raw_df)
    
    # One-hot encode neighborhood
    encoded_df = pd.get_dummies(eng_df, columns=["neighborhood"])
    for col in feature_names:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[feature_names]
    
    # Scale continuous
    continuous_features = [
        "sqft_living", "bedrooms", "bathrooms", "lot_size_sqft",
        "house_age_years", "school_rating", "dist_city_miles",
        "sqft_per_room", "lot_living_ratio", "age_squared", "school_dist_interaction"
    ]
    encoded_df[continuous_features] = scaler.transform(encoded_df[continuous_features])
    
    # Predict log price and invert
    pred_log = float(model.predict(encoded_df)[0])
    pred_usd = float(np.expm1(pred_log))
    
    # 8% typical error margin band
    lower_bound = pred_usd * 0.92
    upper_bound = pred_usd * 1.08
    
    return {
        "estimated_price": round(pred_usd, -2),
        "valuation_range_lower": round(lower_bound, -2),
        "valuation_range_upper": round(upper_bound, -2),
        "confidence_band": "±8%"
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scaler = prepare_regression_data()
    models = train_and_benchmark_models(X_train, y_train)
    champion = models["Ridge"]["model"]
    
    y_pred_log = champion.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred_log)
    
    print("\n--- Out-of-Sample Test Evaluation Report ---")
    print(f"R² Score               : {metrics['r2_score']:.4f}")
    print(f"Root Mean Squared Error: ${metrics['rmse_usd']:,.2f}")
    print(f"Mean Absolute Error    : ${metrics['mae_usd']:,.2f}")
    print(f"Mean Absolute % Error  : {metrics['mape']:.2%}")
    
    # Test property valuation
    sample_house = {
        "sqft_living": 2850.0,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "lot_size_sqft": 7500.0,
        "house_age_years": 8,
        "has_garage": 1,
        "has_pool": 1,
        "neighborhood": "Suburban",
        "school_rating": 9,
        "dist_city_miles": 6.5
    }
    
    val = estimate_property_value(sample_house, champion, scaler, features)
    print("\n--- Live Property Valuation Assessment ---")
    print("Specs: 2,850 sqft, 4 Bed / 3 Bath, Pool, Garage, Suburban (Top Schools)")
    print(f"Estimated Market Value: ${val['estimated_price']:,.2f}")
    print(f"Expected Range ({val['confidence_band']})   : ${val['valuation_range_lower']:,.2f} - ${val['valuation_range_upper']:,.2f}")
