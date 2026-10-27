import pandas as pd

# Day 18: Pandas Missing Data and Cleaning

data = {
    "Student": ["Asha", "Rahul", "Meena", "Arjun", "Priya", "Kiran"],
    "Math": [88, None, 95, 62, 84, 91],
    "Science": [92, 85, None, 70, 89, 87],
    "English": [81, 79, 94, None, 90, 83]
}
df = pd.DataFrame(data)

print("=" * 65)
print("          DAY 18: PANDAS MISSING DATA AND CLEANING")
print("=" * 65)

print("\n1. ORIGINAL DATA")
print(df)

print("\n2. MISSING VALUES PER COLUMN")
print(df.isna().sum())

print("\n3. TOTAL MISSING CELLS")
print(df.isna().sum().sum())

print("\n4. ROWS WITH MISSING VALUES")
print(df[df.isna().any(axis=1)])

# Work on a copy so the original data remains unchanged.
cleaned = df.copy()
subject_columns = ["Math", "Science", "English"]

print("\n5. MISSING MARKS FILLED WITH COLUMN MEANS")
for column in subject_columns:
    cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
print(cleaned.round(2))

print("\n6. MISSING VALUES AFTER FILLING")
print(cleaned.isna().sum())

print("\n7. ROWS AFTER dropna()")
dropped_rows = df.dropna()
print(dropped_rows)

print("\n8. DATASET SIZE COMPARISON")
print("Original shape:", df.shape)
print("After filling shape:", cleaned.shape)
print("After dropping incomplete rows:", dropped_rows.shape)

print("\n9. FLAG ROWS THAT HAD MISSING MARKS")
df_with_flag = df.copy()
df_with_flag["Has_Missing"] = df_with_flag[subject_columns].isna().any(axis=1)
print(df_with_flag)

print("\nDay 18 complete: you checked and handled missing data.")
