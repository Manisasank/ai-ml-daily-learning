import numpy as np

# Creating a NumPy array
data = np.array([10, 20, 30, 40, 50])

print("Array:", data)
print("Shape:", data.shape)
print("Mean:", np.mean(data))
print("Maximum:", np.max(data))
print("Minimum:", np.min(data))
print("Standard Deviation:", np.std(data))

# Basic mathematical operations
print("Add 10:", data + 10)
print("Multiply by 2:", data * 2)

# Finding values greater than 25
print("Values greater than 25:", data[data > 25])