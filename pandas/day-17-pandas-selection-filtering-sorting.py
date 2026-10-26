import pandas as pd

# Day 17: Pandas Selection, Filtering, and Sorting

data = {
    "Student": ["Asha", "Rahul", "Meena", "Arjun", "Priya", "Kiran"],
    "Math": [88, 76, 95, 62, 84, 91],
    "Science": [92, 85, 91, 70, 89, 87],
    "English": [81, 79, 94, 68, 90, 83]
}
df = pd.DataFrame(data)

print("=" * 65)
print("       DAY 17: PANDAS SELECTION, FILTERING, AND SORTING")
print("=" * 65)

print("\n1. COMPLETE DATAFRAME")
print(df)

print("\n2. SELECT ONE COLUMN")
print(df["Student"])

print("\n3. SELECT MULTIPLE COLUMNS")
print(df[["Student", "Math", "Science"]])

print("\n4. SELECT ROWS WITH iloc")
print("First row:")
print(df.iloc[0])
print("Last two rows:")
print(df.iloc[-2:])

print("\n5. FILTER: MATH >= 85")
math_mask = df["Math"] >= 85
print(df.loc[math_mask, ["Student", "Math"]])

print("\n6. FILTER: MATH >= 85 AND SCIENCE >= 90")
both_mask = (df["Math"] >= 85) & (df["Science"] >= 90)
print(df.loc[both_mask, ["Student", "Math", "Science"]])

print("\n7. FILTER: MATH < 70 OR ENGLISH >= 90")
either_mask = (df["Math"] < 70) | (df["English"] >= 90)
print(df.loc[either_mask, ["Student", "Math", "English"]])

print("\n8. FILTER: MATH >= 80 USING NOT")
below_80 = df["Math"] < 80
print(df.loc[~below_80, ["Student", "Math"]])

print("\n9. SORT BY MATH: LOWEST TO HIGHEST")
print(df.sort_values(by="Math"))

print("\n10. SORT BY MATH: HIGHEST TO LOWEST")
print(df.sort_values(by="Math", ascending=False))

print("\n11. SORT BY SCIENCE THEN MATH")
print(df.sort_values(by=["Science", "Math"], ascending=[False, False]))

print("\n12. FILTER AND RESET INDEX")
top_students = df.loc[df["Math"] >= 85].copy()
top_students = top_students.reset_index(drop=True)
print(top_students)

print("\nDay 17 complete: you can select, filter, and sort DataFrames.")
