
import numpy as np

# Day 6: NumPy Reshaping and Transposing

# 1. Create a 1D array
numbers = np.array([1, 2, 3, 4, 5, 6])

print("Original Array:", numbers)
print("Original Shape:", numbers.shape)

# 2. Reshape into 2 rows and 3 columns
matrix = numbers.reshape(2, 3)

print("\nReshaped Array:")
print(matrix)
print("Shape:", matrix.shape)

# 3. Let NumPy calculate one dimension
print("\nReshape using -1:")
print(numbers.reshape(3, -1))

# 4. Flatten a 2D array
print("\nFlattened Array:")
print(matrix.flatten())

# 5. Transpose rows and columns
print("\nTransposed Matrix:")
print(matrix.T)
print("Transposed Shape:", matrix.T.shape)

# 6. Add row and column dimensions
values = np.array([10, 20, 30])

row = values[np.newaxis, :]
column = values[:, np.newaxis]

print("\nRow Array:")
print(row)
print("Row Shape:", row.shape)

print("\nColumn Array:")
print(column)
print("Column Shape:", column.shape)

# 7. AI/ML example: two samples, three features each
features = np.array([1, 2, 3, 4, 5, 6])
batch = features.reshape(2, 3)

print("\nFeature Batch:")
print(batch)
print("Batch Shape:", batch.shape)
