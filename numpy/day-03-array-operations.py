import numpy as np

# Creating arrays
a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

print("Array A:", a)
print("Array B:", b)

# Arithmetic operations
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)

# Power
print("Power:", a ** 2)

# Aggregation operations
print("Sum:", np.sum(a))
print("Mean:", np.mean(a))
print("Minimum:", np.min(a))
print("Maximum:", np.max(a))
print("Standard Deviation:", np.std(a))

# Square root
print("Square Root:", np.sqrt(a))

# Comparison
print("A greater than B:", a > b)

# Concatenation
combined = np.concatenate((a, b))
print("Concatenated Array:", combined)