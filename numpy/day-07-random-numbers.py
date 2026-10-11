
import numpy as np

# Day 7: NumPy Random Numbers and Reproducibility

# 1. Create a random generator with a fixed seed
rng = np.random.default_rng(42)

# 2. Generate random integers from 1 through 10
integers = rng.integers(1, 11, size=5)
print("Random Integers:", integers)

# 3. Generate random decimal values from 0 to less than 1
decimals = rng.random(5)
print("\nRandom Decimals:", decimals)

# 4. Generate a random 2D array
matrix = rng.integers(0, 100, size=(3, 4))
print("\nRandom Matrix:")
print(matrix)
print("Matrix Shape:", matrix.shape)

# 5. Select random topics without replacement
topics = np.array(["Python", "NumPy", "Pandas", "Matplotlib"])
selected = rng.choice(topics, size=2, replace=False)
print("\nSelected Topics:", selected)

# 6. Shuffle an array without changing the original
numbers = np.array([1, 2, 3, 4, 5, 6])
shuffled = rng.permutation(numbers)
print("\nOriginal Array:", numbers)
print("Shuffled Array:", shuffled)

# 7. Generate values from a normal distribution
normal_values = rng.normal(loc=0, scale=1, size=5)
print("\nNormal Distribution Samples:", normal_values)

# 8. Demonstrate reproducibility
rng_a = np.random.default_rng(7)
rng_b = np.random.default_rng(7)

result_a = rng_a.integers(1, 100, size=5)
result_b = rng_b.integers(1, 100, size=5)

print("\nGenerator A:", result_a)
print("Generator B:", result_b)
print("Same results:", np.array_equal(result_a, result_b))
