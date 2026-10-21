import numpy as np

# Day 15: NumPy Practice and Revision

students = np.array(["Asha", "Rahul", "Meena", "Arjun", "Priya"])
subjects = np.array(["Math", "Science", "English"])
marks = np.array([
    [88, 92, 81],
    [76, 85, 79],
    [95, 91, 94],
    [62, 70, 68],
    [84, 89, 90]
])

print("=" * 60)
print("             NUMPY PRACTICE AND REVISION")
print("=" * 60)

# 1. Inspect the dataset.
print("\n1. DATASET")
print("Students:", students)
print("Subjects:", subjects)
print(marks)
print("Shape (students, subjects):", marks.shape)

# 2. Indexing and slicing.
print("\n2. INDEXING")
print("Meena's marks:", marks[2])
print("All students' Math marks:", marks[:, 0])

# 3. Student totals and averages. axis=1 works across rows.
totals = np.sum(marks, axis=1)
averages = np.mean(marks, axis=1)
print("\n3. STUDENT TOTALS AND AVERAGES")
for i in range(len(students)):
    print(f"{students[i]:<8} Total: {totals[i]:>3}  Average: {averages[i]:.2f}")

# 4. Subject averages. axis=0 works down columns.
subject_averages = np.mean(marks, axis=0)
print("\n4. SUBJECT AVERAGES")
for i in range(len(subjects)):
    print(f"{subjects[i]:<8}: {subject_averages[i]:.2f}")

# 5. Find the top student.
top_index = np.argmax(totals)
print("\n5. TOP STUDENT")
print(f"{students[top_index]} with {totals[top_index]} total marks")

# 6. Rank students from highest total to lowest.
ranking = np.argsort(totals)[::-1]
print("\n6. RANKING")
for position, index in enumerate(ranking, start=1):
    print(f"{position}. {students[index]} - {totals[index]}")

# 7. Filter students with averages at least 85.
high_mask = averages >= 85
print("\n7. STUDENTS WITH AVERAGE >= 85")
print(students[high_mask])

# 8. Find students with any subject mark below 70.
support_mask = np.any(marks < 70, axis=1)
print("\n8. STUDENTS WITH ANY MARK BELOW 70")
print(students[support_mask])

# 9. Count high marks and find extremes.
print("\n9. MARK COUNTS AND EXTREMES")
print("Number of marks >= 90:", np.sum(marks >= 90))
print("Highest mark:", np.max(marks))
print("Lowest mark:", np.min(marks))

# 10. Sort each student's marks from low to high.
print("\n10. MARKS SORTED WITHIN EACH STUDENT")
print(np.sort(marks, axis=1))

# 11. Get row/column positions of marks at least 90.
positions = np.argwhere(marks >= 90)
print("\n11. POSITIONS OF MARKS >= 90 (row, column)")
print(positions)

# 12. Change a copy, preserving the original.
cleaned_marks = marks.copy()
cleaned_marks[cleaned_marks < 70] = 70
print("\n12. SAFE UPDATE")
print("Updated copy:")
print(cleaned_marks)
print("Original data remains unchanged:")
print(marks)

# 13. Overall class statistics.
print("\n13. CLASS STATISTICS")
print(f"Overall mean: {np.mean(marks):.2f}")
print(f"Overall median: {np.median(marks):.2f}")
print(f"Overall standard deviation: {np.std(marks):.2f}")

print("\nDay 15 revision complete.")
