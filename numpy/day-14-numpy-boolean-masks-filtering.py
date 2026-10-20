import numpy as np

# Day 14: NumPy Boolean Masks and Filtering

students = np.array(["Asha", "Rahul", "Meena", "Arjun", "Priya"])
marks = np.array([88, 76, 95, 62, 84])

print("=" * 55)
print("       NUMPY BOOLEAN MASKS AND FILTERING")
print("=" * 55)

# 1. Create a Boolean mask for marks greater than or equal to 80.
passed_mask = marks >= 80
print("\n1. Boolean mask (marks >= 80):")
print(passed_mask)

# 2. Use the mask to filter marks.
high_marks = marks[passed_mask]
print("\n2. Marks greater than or equal to 80:")
print(high_marks)

# 3. Use the same mask to filter student names.
high_scoring_students = students[passed_mask]
print("\n3. Students scoring at least 80:")
print(high_scoring_students)

# 4. Filter marks below 70.
low_mask = marks < 70
print("\n4. Boolean mask (marks < 70):")
print(low_mask)
print("Marks below 70:", marks[low_mask])
print("Students with marks below 70:", students[low_mask])

# 5. Combine conditions with AND (&).
middle_mask = (marks >= 70) & (marks < 90)
print("\n5. Marks from 70 up to, but not including, 90:")
print(marks[middle_mask])
print("Matching students:", students[middle_mask])

# 6. Combine conditions with OR (|).
extreme_mask = (marks < 70) | (marks >= 90)
print("\n6. Marks below 70 OR at least 90:")
print(marks[extreme_mask])
print("Matching students:", students[extreme_mask])

# 7. Reverse a condition with NOT (~).
not_high_mask = ~(marks >= 80)
print("\n7. Marks that are NOT at least 80:")
print(marks[not_high_mask])
print("Matching students:", students[not_high_mask])

# 8. Count how many students meet a condition.
number_high = np.sum(marks >= 80)
print("\n8. Number of students scoring at least 80:")
print(number_high)

# 9. Safely update values using a mask.
adjusted_marks = marks.copy()
adjusted_marks[adjusted_marks < 70] = 70
print("\n9. Marks after raising scores below 70 to 70 (copy only):")
print(adjusted_marks)
print("Original marks remain unchanged:", marks)

print("\nFiltering practice complete.")
