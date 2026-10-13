import numpy as np

# Day 9: NumPy Boolean Masks and Filtering

# 1. Create an array
marks = np.array([78, 92, 65, 92, 48, 85, 65, 100, 73])
print("Marks:", marks)

# 2. Create a Boolean mask
passed_mask = marks >= 40
print("\nPassed mask:", passed_mask)
print("Passed marks:", marks[passed_mask])

# 3. Filter marks using a condition
high_marks = marks[marks >= 80]
print("\nMarks >= 80:", high_marks)

# 4. Combine conditions: AND (&)
medium_marks = marks[(marks >= 60) & (marks < 90)]
print("Marks from 60 to below 90:", medium_marks)

# 5. Combine conditions: OR (|)
special_marks = marks[(marks < 50) | (marks >= 95)]
print("Marks below 50 or >= 95:", special_marks)

# 6. Use NOT (~) with a Boolean mask
not_high_mask = ~(marks >= 90)
print("\nMarks below 90:", marks[not_high_mask])

# 7. Count matching values
print("Number of marks >= 80:", np.sum(marks >= 80))
print("Any mark below 35?", np.any(marks < 35))
print("All marks non-negative?", np.all(marks >= 0))

# 8. Filter a 2D array
scores = np.array([
    [78, 92, 65],
    [48, 85, 100],
    [73, 55, 89]
])
print("\n2D scores:")
print(scores)
print("Scores >= 80:", scores[scores >= 80])

# 9. Replace values using a mask
updated_marks = marks.copy()
updated_marks[updated_marks < 40] = 40
print("\nMarks after setting values below 40 to 40:", updated_marks)

# 10. Use np.where to create a result from a condition
status = np.where(marks >= 40, "Pass", "Fail")
print("\nStatus:", status)

# 11. AI/ML example: select predictions above a confidence threshold
confidence = np.array([0.42, 0.91, 0.76, 0.63, 0.98])
confident_predictions = confidence[confidence >= 0.75]
print("Confident predictions (>= 0.75):", confident_predictions)
