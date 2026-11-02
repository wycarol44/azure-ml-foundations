import pandas as pd

v1 = pd.read_csv("data/telco_churn.csv")

v2 = pd.read_csv("data/telco_churn_v2.csv")

print("Version 1 rows:", len(v1))
print("Version 2 rows:", len(v2))

print("\nVersion 1 columns:")
print(v1.columns.tolist())

print("\nVersion 2 columns:")
print(v2.columns.tolist())

print("\nSchema comparison:")

for col in v1.columns:
    print(
        col,
        "V1:", v1[col].dtype,
        "V2:", v2[col].dtype if col in v2 else "Missing"
    )