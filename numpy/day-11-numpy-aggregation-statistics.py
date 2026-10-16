import numpy as np

# Day 11: NumPy Aggregation and Statistical Summaries

marks = np.array([78, 92, 65, 92, 48, 85, 65, 100, 73])
print("Marks:", marks)

# Basic aggregation
print("\nTotal:", np.sum(marks))
print("Mean:", np.mean(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Count:", np.size(marks))

# Median and percentiles
print("\nMedian:", np.median(marks))
print("25th percentile:", np.percentile(marks, 25))
print("75th percentile:", np.percentile(marks, 75))

# Measures of spread
print("\nVariance:", np.var(marks))
print("Standard deviation:", np.std(marks))
print("Range:", np.max(marks) - np.min(marks))

# Boolean aggregation
print("\nMarks >= 80:", np.sum(marks >= 80))
print("Any mark below 35?", np.any(marks < 35))
print("All marks between 0 and 100?", np.all((marks >= 0) & (marks <= 100)))

# Aggregation along axes
scores = np.array([[78, 92, 65], [48, 85, 100], [73, 55, 89]])
print("\nScores:")
print(scores)
print("Row totals (axis=1):", np.sum(scores, axis=1))
print("Column totals (axis=0):", np.sum(scores, axis=0))
print("Column means:", np.mean(scores, axis=0))

# Cumulative sums and products
values = np.array([2, 3, 4, 5])
print("\nCumulative sum:", np.cumsum(values))
print("Cumulative product:", np.cumprod(values))

# Ignore missing values when appropriate
measurements = np.array([10.0, 20.0, np.nan, 40.0])
print("\nMean ignoring NaN:", np.nanmean(measurements))
print("Finite measurement count:", np.isfinite(measurements).sum())

# AI/ML example: model confidence
confidence = np.array([0.72, 0.91, 0.84, 0.66, 0.95])
print("\nAverage confidence:", np.mean(confidence))
print("Highest confidence:", np.max(confidence))
print("Predictions >= 0.80:", np.sum(confidence >= 0.80))
