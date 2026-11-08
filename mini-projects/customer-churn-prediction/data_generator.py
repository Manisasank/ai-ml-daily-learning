"""
Customer Churn Dataset Generator
Generates a realistic synthetic dataset simulating customer demographics,
account information, usage patterns, and churn behavior.
"""

import os
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.
    
    Parameters:
        n_samples: Number of customer records to generate.
        random_state: Seed for reproducibility.
        
    Returns:
        pd.DataFrame containing customer profiles and churn status.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 73, size=n_samples)
    age = np.random.randint(18, 75, size=n_samples)
    
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_methods = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.40, 0.60])
    
    monthly_charges = np.round(np.random.uniform(20.0, 115.0, size=n_samples), 2)
    # Total charges loosely correlated with tenure and monthly charges plus noise
    total_charges = np.round(tenure_months * monthly_charges * np.random.uniform(0.95, 1.05, size=n_samples), 2)
    
    # Calculate churn probability based on realistic business risk factors
    log_odds = (
        -1.5
        + (monthly_charges > 70) * 0.8
        + (contract_types == "Month-to-month") * 1.2
        - (contract_types == "Two year") * 1.5
        - (tenure_months > 24) * 0.9
        + (payment_methods == "Electronic check") * 0.5
        - (tech_support == "Yes") * 0.6
        - (online_security == "Yes") * 0.4
    )
    
    probabilities = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "tenure_months": tenure_months,
        "contract": contract_types,
        "payment_method": payment_methods,
        "tech_support": tech_support,
        "online_security": online_security,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn
    })
    
    return df


if __name__ == "__main__":
    df = generate_churn_dataset(2500)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_churn_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customer records with churn rate: {df['churn'].mean():.2%}")
    print(df.head())
