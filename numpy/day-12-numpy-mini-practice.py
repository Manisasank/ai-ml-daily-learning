import numpy as np

# Day 12: NumPy Mini Practice
# Review sorting, aggregation, Boolean masks, and missing values

scores = np.array([78, 92, 65, 92, 48, 85, 65, 100, 73])

# 1. Sort scores
print("Original scores:", scores)
print("Ascending:", np.sort(scores))
print("Descending:", np.sort(scores)[::-1])

# 2. Summarize the scores
print("\nSum:", np.sum(scores))
print("Mean:", np.mean(scores))
print("Median:", np.median(scores))
print("Minimum:", np.min(scores))
print("Maximum:", np.max(scores))

# 3. Find positions of lowest and highest scores
print("\nMinimum index:", np.argmin(scores))
print("Maximum index:", np.argmax(scores))

# 4. Filter scores using Boolean masks
print("\nScores >= 75:", scores[scores >= 75])
print("Scores from 60 to 89:", scores[(scores >= 60) & (scores <= 89)])
print("Scores below 50 or equal to 100:", scores[(scores < 50) | (scores == 100)])

# 5. Count matching values and find unique values
print("\nNumber of 92s:", np.sum(scores == 92))
print("Indices of 92:", np.where(scores == 92)[0])
unique_scores, counts = np.unique(scores, return_counts=True)
print("Unique scores:", unique_scores)
print("Counts:", counts)

# 6. Aggregate a 2D marks table
marks = np.array([
    [78, 92, 65],
    [48, 85, 100],
    [73, 55, 89]
])
print("\nMarks table:")
print(marks)
print("Row totals:", np.sum(marks, axis=1))
print("Column totals:", np.sum(marks, axis=0))
print("Row means:", np.mean(marks, axis=1))
print("Column means:", np.mean(marks, axis=0))

# 7. Handle missing and non-finite values
readings = np.array([22.5, np.nan, 23.0, np.inf, 21.8, np.nan, 24.1])
finite_readings = readings[np.isfinite(readings)]
print("\nFinite readings:", finite_readings)
print("Missing-value count:", np.isnan(readings).sum())
print("Mean of finite readings:", np.mean(finite_readings))

# 8. Replace low scores in a copy, preserving the original
adjusted_scores = scores.copy()
adjusted_scores[adjusted_scores < 60] = 60
print("\nOriginal unchanged:", scores)
print("Adjusted copy:", adjusted_scores)
