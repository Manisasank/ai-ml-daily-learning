import numpy as np

# Day 13: NumPy Student Marks Analyzer

students = np.array(["Asha", "Rahul", "Meena", "Arjun", "Priya"])
subjects = np.array(["Math", "Science", "English"])

# Each row is a student; each column is a subject.
marks = np.array([
    [88, 92, 81],
    [76, 85, 79],
    [95, 91, 94],
    [62, 70, 68],
    [84, 89, 90]
])

print("=" * 55)
print("           STUDENT MARKS ANALYZER")
print("=" * 55)
print("\nMarks table (rows = students, columns = subjects):")
print(marks)

# 1. Student totals and averages (axis=1 means across each row).
totals = np.sum(marks, axis=1)
averages = np.mean(marks, axis=1)

print("\n1. STUDENT TOTALS AND AVERAGES")
for i in range(len(students)):
    print(f"{students[i]:<8} Total: {totals[i]:>3} / {len(subjects) * 100}"
          f"   Average: {averages[i]:.2f}")

# 2. Subject averages (axis=0 means down each column).
subject_averages = np.mean(marks, axis=0)
print("\n2. SUBJECT AVERAGES")
for i in range(len(subjects)):
    print(f"{subjects[i]:<8}: {subject_averages[i]:.2f}")

# 3. Top student.
top_index = np.argmax(totals)
print("\n3. TOP STUDENT")
print(f"{students[top_index]} with {totals[top_index]} marks "
      f"(average {averages[top_index]:.2f})")

# 4. Ranking from highest total to lowest.
ranking = np.argsort(totals)[::-1]
print("\n4. RANKING (highest total first)")
for position, index in enumerate(ranking, start=1):
    print(f"{position}. {students[index]} - {totals[index]} marks")

# 5. Students whose average is at least 85.
high_performers = students[averages >= 85]
print("\n5. STUDENTS WITH AVERAGE >= 85")
print(high_performers if high_performers.size else "No students meet this condition.")

# 6. Students with a mark below 70 in any subject.
needs_support = students[np.any(marks < 70, axis=1)]
print("\n6. STUDENTS WITH A MARK BELOW 70 IN ANY SUBJECT")
print(needs_support if needs_support.size else "No students meet this condition.")

# 7. Highest and lowest individual marks.
highest_mark = np.max(marks)
lowest_mark = np.min(marks)
highest_position = np.argwhere(marks == highest_mark)[0]
lowest_position = np.argwhere(marks == lowest_mark)[0]

print("\n7. HIGHEST AND LOWEST INDIVIDUAL MARKS")
print(f"Highest: {highest_mark} - {students[highest_position[0]]}, "
      f"{subjects[highest_position[1]]}")
print(f"Lowest:  {lowest_mark} - {students[lowest_position[0]]}, "
      f"{subjects[lowest_position[1]]}")

# 8. Overall class statistics.
print("\n8. CLASS STATISTICS")
print(f"Class average: {np.mean(marks):.2f}")
print(f"Class median:  {np.median(marks):.2f}")
print(f"Class highest: {np.max(marks)}")
print(f"Class lowest:  {np.min(marks)}")

print("\nAnalysis complete.")
