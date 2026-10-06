import numpy as np

# 1D array
data = np.array([10, 20, 30, 40, 50])

print("Original Array:", data)

# Indexing
print("First Element:", data[0])
print("Last Element:", data[-1])

# Slicing
print("Elements from index 1 to 3:", data[1:4])
print("First Three Elements:", data[:3])
print("From Index 2:", data[2:])

# Step
print("Every Second Element:", data[::2])

# Reverse
print("Reversed Array:", data[::-1])

# Modifying an element
data[2] = 100
print("Modified Array:", data)

# 2D array
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("\n2D Array:")
print(matrix)

# Shape
print("Shape:", matrix.shape)

# Accessing elements
print("Element at Row 1, Column 2:", matrix[1, 2])

# Rows
print("First Row:", matrix[0])

# Columns
print("First Column:", matrix[:, 0])

# Reshape
numbers = np.array([1, 2, 3, 4, 5, 6])
reshaped = numbers.reshape(2, 3)

print("\nReshaped Array:")
print(reshaped)

# Filtering
print("Values Greater Than 50:", matrix[matrix > 50])