import pandas as pd

df = pd.read_csv("data/telco_churn.csv")

# Clean whitespace in TotalCharges.
df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()

# Convert to numeric.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove records with invalid TotalCharges.
df_v2 = df.dropna(subset=["TotalCharges"]).copy()

df_v2.to_csv(
    "data/telco_churn_v2.csv",
    index=False
)

print("Version 1 shape:", df.shape)
print("Version 2 shape:", df_v2.shape)
print("Version 2 missing values:")
print(df_v2.isnull().sum())