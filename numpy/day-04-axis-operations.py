import numpy as np

# Creating a 2D array
data = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Array:")
print(data)

# Total sum
print("Total Sum:", np.sum(data))

# Column-wise operations
print("Column-wise Sum:", np.sum(data, axis=0))
print("Column-wise Mean:", np.mean(data, axis=0))
print("Column-wise Minimum:", np.min(data, axis=0))
print("Column-wise Maximum:", np.max(data, axis=0))

# Row-wise operations
print("Row-wise Sum:", np.sum(data, axis=1))
print("Row-wise Mean:", np.mean(data, axis=1))
print("Row-wise Minimum:", np.min(data, axis=1))
print("Row-wise Maximum:", np.max(data, axis=1))