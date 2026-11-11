"""
Real Estate Housing Dataset Generator
Synthesizes a realistic multivariate dataset of property characteristics,
neighborhood ratings, architectural attributes, and market selling prices.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_housing_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic housing transaction data with realistic market dynamics.
    
    Parameters:
        n_samples: Total number of house records.
        random_state: Reproducibility seed.
        
    Returns:
        pd.DataFrame containing property attributes and sale prices.
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Structural features
    sqft_living = rng.normal(loc=2100, scale=650, size=n_samples)
    sqft_living = np.clip(sqft_living, 600, 5500).round()
    
    bedrooms = np.clip(np.round(sqft_living / 600 + rng.normal(0, 0.6, n_samples)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.75 + rng.normal(0, 0.4, n_samples) * 0.5, 1), 1.0, 5.0)
    
    lot_size_sqft = np.clip(sqft_living * rng.uniform(1.5, 4.0, n_samples), 1000, 25000).round()
    house_age_years = rng.randint(0, 65, size=n_samples)
    has_garage = rng.choice([1, 0], size=n_samples, p=[0.75, 0.25])
    has_pool = rng.choice([1, 0], size=n_samples, p=[0.20, 0.80])
    
    # 2. Location & Neighborhood features
    neighborhood_tier = rng.choice(["Suburban", "Urban", "Luxury Waterfront", "Rural"], size=n_samples, p=[0.45, 0.30, 0.15, 0.10])
    school_rating = rng.randint(1, 11, size=n_samples)  # 1 to 10
    distance_to_city_center_miles = np.clip(rng.exponential(scale=12, size=n_samples), 1.0, 45.0).round(1)
    
    # Tier price multipliers
    tier_multipliers = {
        "Suburban": 1.0,
        "Urban": 1.25,
        "Luxury Waterfront": 1.75,
        "Rural": 0.80
    }
    tier_effect = np.array([tier_multipliers[t] for t in neighborhood_tier])
    
    # 3. Base Price calculation with non-linear relationships and noise
    base_price_per_sqft = 160.0
    
    price = (
        (sqft_living * base_price_per_sqft)
        + (bedrooms * 15000)
        + (bathrooms * 22000)
        + (has_garage * 25000)
        + (has_pool * 45000)
        + (school_rating * 12000)
        - (distance_to_city_center_miles * 2800)
        - (house_age_years * 1200)
    )
    
    # Apply neighborhood multiplier
    price = price * tier_effect
    
    # Add market variance / noise (normal distribution)
    noise = rng.normal(0, 25000, n_samples)
    final_price = np.clip(price + noise, 80000, 2500000).round(-2)
    
    df = pd.DataFrame({
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size_sqft": lot_size_sqft,
        "house_age_years": house_age_years,
        "has_garage": has_garage,
        "has_pool": has_pool,
        "neighborhood": neighborhood_tier,
        "school_rating": school_rating,
        "dist_city_miles": distance_to_city_center_miles,
        "price": final_price
    })
    
    return df


if __name__ == "__main__":
    df = generate_housing_data(n_samples=2000)
    print(f"Generated {len(df)} property records.")
    print("\nSummary Statistics of Housing Prices:")
    print(df["price"].describe().apply(lambda x: f"${x:,.2f}"))
    print("\nSample records:")
    print(df.head(3))
