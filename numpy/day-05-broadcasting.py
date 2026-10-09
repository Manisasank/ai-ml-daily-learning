
import numpy as np

# Day 5: NumPy Broadcasting

# 1. Scalar Broadcasting
arr = np.array([10, 20, 30])

print("Original Array:", arr)
print("Add 5:", arr + 5)
print("Multiply by 2:", arr * 2)

# 2. Broadcasting with a 2D Array
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("\nOriginal Matrix:")
print(matrix)

print("Matrix + 5:")
print(matrix + 5)

# 3. Broadcasting Two Arrays
a = np.array([10, 20, 30])
b = np.array([1])

print("\nArray A:", a)
print("Array B:", b)
print("A + B:", a + b)

# 4. Row-wise Broadcasting
bonus = np.array([1, 2, 3])

print("\nRow-wise Broadcasting:")
print(matrix + bonus)

# 5. Column-wise Broadcasting
column_bonus = np.array([
    [5],
    [10]
])

print("\nColumn-wise Broadcasting:")
print(matrix + column_bonus)

# 6. AI/ML Example: Mean-Centering
data = np.array([
    [10, 100],
    [20, 200],
    [30, 300]
])

feature_means = np.mean(data, axis=0)
centered_data = data - feature_means

print("\nFeature Means:", feature_means)
print("Mean-Centered Data:")
print(centered_data)
