import pandas as pd

# Day 16: Pandas Series and DataFrame Basics

print("=" * 60)
print("             DAY 16: PANDAS BASICS")
print("=" * 60)

# 1. Create a Series from a list.
marks = pd.Series([88, 76, 95, 62, 84])
print("\n1. SERIES OF MARKS")
print(marks)
print("First mark:", marks.iloc[0])
print("Number of marks:", len(marks))

# 2. Create a Series with custom labels.
student_marks = pd.Series(
    [88, 76, 95],
    index=["Asha", "Rahul", "Meena"]
)
print("\n2. SERIES WITH STUDENT LABELS")
print(student_marks)
print("Meena's mark:", student_marks.loc["Meena"])

# 3. Create a DataFrame from a dictionary.
data = {
    "Student": ["Asha", "Rahul", "Meena", "Arjun", "Priya"],
    "Math": [88, 76, 95, 62, 84],
    "Science": [92, 85, 91, 70, 89],
    "English": [81, 79, 94, 68, 90]
}
df = pd.DataFrame(data)

print("\n3. STUDENT DATAFRAME")
print(df)

# 4. Inspect the table.
print("\n4. DATAFRAME INFORMATION")
print("Shape (rows, columns):", df.shape)
print("Column names:", list(df.columns))
print("First two rows:")
print(df.head(2))
print("Data types:")
print(df.dtypes)

# 5. Select one column (returns a Series).
print("\n5. SELECT ONE COLUMN")
print(df["Student"])
print("Math marks:")
print(df["Math"])

# 6. Select multiple columns (returns a DataFrame).
print("\n6. SELECT MULTIPLE COLUMNS")
print(df[["Student", "Math", "Science"]])

# 7. Select rows by position with iloc.
print("\n7. SELECT ROWS BY POSITION (iloc)")
print("First row:")
print(df.iloc[0])
print("Rows 1 through 3 (stop position 3 is excluded):")
print(df.iloc[1:3])

# 8. Select rows by condition with loc.
print("\n8. SELECT ROWS BY CONDITION (loc)")
high_math = df.loc[df["Math"] >= 85, ["Student", "Math"]]
print("Students scoring at least 85 in Math:")
print(high_math)

# 9. Add calculated columns.
df["Total"] = df[["Math", "Science", "English"]].sum(axis=1)
df["Average"] = df[["Math", "Science", "English"]].mean(axis=1).round(2)
print("\n9. ADD TOTAL AND AVERAGE COLUMNS")
print(df)

# 10. Basic summary statistics.
print("\n10. SUMMARY STATISTICS")
print("Average Math mark:", round(df["Math"].mean(), 2))
print("Highest Science mark:", df["Science"].max())
print("Lowest English mark:", df["English"].min())
print("Number of students:", df["Student"].count())

print("\nDay 16 complete: you created and explored Series and DataFrames.")
