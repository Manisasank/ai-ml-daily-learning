
import numpy as np

# Day 8: NumPy Sorting, Searching, and Unique Values

# 1. Sort arrays
values = np.array([40, 10, 30, 20, 10])

print("Original:", values)
print("Sorted:", np.sort(values))
print("Descending:", np.sort(values)[::-1])

# 2. Sort each row
matrix = np.array([
    [30, 10, 20],
    [60, 40, 50]
])

print("\nSorted Rows:")
print(np.sort(matrix, axis=1))

# 3. Search for a value
numbers = np.array([10, 20, 30, 20, 40])

print("\nIndices of 20:", np.where(numbers == 20)[0])

# 4. Filter values
data = np.array([12, 45, 7, 60, 33])

print("Values greater than 30:", data[data > 30])

# 5. Find unique values and counts
items = np.array([10, 20, 10, 30, 20, 10])

unique_values, counts = np.unique(
    items, return_counts=True
)

print("\nUnique Values:", unique_values)
print("Counts:", counts)

# 6. Minimum and maximum positions
scores = np.array([35, 12, 87, 24, 65])

print("\nMinimum:", np.min(scores))
print("Minimum Index:", np.argmin(scores))
print("Maximum:", np.max(scores))
print("Maximum Index:", np.argmax(scores))
