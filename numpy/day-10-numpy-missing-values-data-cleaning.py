import numpy as np

# Day 10: NumPy Missing Values and Basic Data Cleaning

temperatures = np.array([24.5, 25.0, np.nan, 26.2, 24.8, np.nan])
print("Temperatures:", temperatures)

# Detect missing values
missing_mask = np.isnan(temperatures)
print("\nMissing-value mask:", missing_mask)
print("Number of missing values:", np.sum(missing_mask))

# Select valid values
valid_temperatures = temperatures[~missing_mask]
print("Valid temperatures:", valid_temperatures)

# Statistics that ignore NaN
print("\nMean ignoring NaN:", np.nanmean(temperatures))
print("Minimum ignoring NaN:", np.nanmin(temperatures))
print("Maximum ignoring NaN:", np.nanmax(temperatures))
print("Sum ignoring NaN:", np.nansum(temperatures))

# Replace NaN with the mean of valid values
mean_temperature = np.nanmean(temperatures)
filled_temperatures = np.where(
    np.isnan(temperatures), mean_temperature, temperatures
)
print("\nFilled temperatures:", filled_temperatures)

# Missing values in a 2D array
data = np.array([
    [10.0, 20.0, np.nan],
    [15.0, np.nan, 35.0],
    [12.0, 22.0, 32.0]
])
print("\nOriginal data:")
print(data)
print("Missing-value count:", np.isnan(data).sum())

# Replace NaN with zero only when zero is meaningful
zero_filled = np.nan_to_num(data, nan=0.0)
print("\nZero-filled data:")
print(zero_filled)

# Detect infinite values as well as NaN
values = np.array([10.0, np.inf, -np.inf, np.nan, 25.0])
print("\nFinite-value mask:", np.isfinite(values))
print("Finite values:", values[np.isfinite(values)])

# Validate numeric data in a range
scores = np.array([85.0, 92.0, np.nan, -5.0, 78.0])
valid_scores_mask = np.isfinite(scores) & (scores >= 0) & (scores <= 100)
print("\nValid scores:", scores[valid_scores_mask])
print("Invalid scores:", scores[~valid_scores_mask])

# AI/ML example: remove records with missing measurements
measurements = np.array([1.2, 1.5, np.nan, 1.7, 1.4])
clean_measurements = measurements[np.isfinite(measurements)]
print("\nClean measurements:", clean_measurements)
