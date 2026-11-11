"""
Model Training & Regularization Selection
Trains Linear Regression, Ridge, Lasso, and Random Forest models
with 5-fold cross validation and coefficient inspection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

try:
    from .preprocessor import prepare_regression_data
except ImportError:
    from preprocessor import prepare_regression_data


def train_and_benchmark_models(X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train regression models with cross-validation on log-price scale.
    """
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=42),
        "Lasso": Lasso(alpha=0.005, max_iter=2000, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }
    
    benchmark = {}
    
    for name, model in candidates.items():
        # Evaluate 5-fold R2
        r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        # Evaluate 5-fold RMSE
        rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        
        model.fit(X_train, y_train)
        
        benchmark[name] = {
            "model": model,
            "cv_r2_mean": float(r2_scores.mean()),
            "cv_r2_std": float(r2_scores.std()),
            "cv_rmse_mean": float(rmse_scores.mean()),
            "cv_rmse_std": float(rmse_scores.std())
        }
        print(f"[{name:16s}] 5-Fold R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f}) | RMSE (log): {rmse_scores.mean():.4f}")
        
    return benchmark


def inspect_feature_importance(model, feature_names, top_n: int = 6):
    """
    Print top positive and negative coefficient drivers for linear/regularized models.
    """
    if hasattr(model, "coef_"):
        coefs = model.coef_
        df_coef = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        df_sorted = df_coef.sort_values(by="coefficient", key=abs, ascending=False)
        print(f"\nTop {top_n} Driving Coefficients:")
        for _, row in df_sorted.head(top_n).iterrows():
            print(f"  {row['feature']:26s} : {row['coefficient']:+.4f}")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        df_sorted = df_imp.sort_values(by="importance", ascending=False)
        print(f"\nTop {top_n} Feature Importances (Tree):")
        for _, row in df_sorted.head(top_n).iterrows():
            print(f"  {row['feature']:26s} : {row['importance']:.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scaler = prepare_regression_data()
    print("Beginning 5-fold cross validation on regression models...\n")
    results = train_and_benchmark_models(X_train, y_train)
    
    # Select best model based on R2
    best_name = max(results.keys(), key=lambda k: results[k]["cv_r2_mean"])
    best_model = results[best_name]["model"]
    print(f"\nChampion Regressor: {best_name} (R² = {results[best_name]['cv_r2_mean']:.4f})")
    
    inspect_feature_importance(best_model, features)
