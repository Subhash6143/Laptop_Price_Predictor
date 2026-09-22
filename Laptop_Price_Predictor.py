import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("laptop.csv")

print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

# This is just the old index from the original dataset.
# It does not contain useful information for predicting price.

df.drop("Unnamed: 0", axis=1, inplace=True)


# ==========================================
# 3. CLEAN RAM
# ==========================================

# Example:
# "8 GB"  -> 8
# "16 GB" -> 16

df["RAM"] = df["RAM"].str.extract(r"(\d+)").astype(float)


# ==========================================
# 4. CLEAN PROCESSOR SPEED
# ==========================================

# Example:
# "4.2 Ghz Processor" -> 4.2
# "2.4 Ghz Processor" -> 2.4

df["Ghz"] = df["Ghz"].str.extract(r"([\d.]+)").astype(float)


# ==========================================
# 5. CLEAN DISPLAY SIZE
# ==========================================

# Example:
# "15.6 Inch" -> 15.6
# "14 Inch"   -> 14

df["Display"] = df["Display"].str.extract(r"([\d.]+)").astype(float)
# Fill missing display size with the median
df["Display"] = df["Display"].fillna(df["Display"].median())

# ==========================================
# 6. FUNCTION TO CONVERT STORAGE
# ==========================================

def convert_storage(value):

    value = str(value).upper().strip()

    # Extract the numerical part
    number = pd.to_numeric(
        pd.Series(value).str.extract(r"([\d.]+)")[0].iloc[0],
        errors="coerce"
    )

    # If no number exists
    if pd.isna(number):
        return 0

    # Convert TB to GB
    if "TB" in value:
        number = number * 1024

    return number


# ==========================================
# 7. CLEAN SSD
# ==========================================

# Examples:
# "512 GB SSD Storage" -> 512
# "1 TB SSD Storage"   -> 1024

df["SSD"] = df["SSD"].apply(convert_storage)


# ==========================================
# 8. CLEAN HDD
# ==========================================

# Examples:
# "No HDD"             -> 0
# "1 TB HDD Storage"   -> 1024
# "512 GB HDD Storage" -> 512

df["HDD"] = df["HDD"].apply(convert_storage)


# ==========================================
# 9. CLEAN ADAPTER
# ==========================================

# Example:
# "45" -> 45
# "65" -> 65
# "no" -> NaN

df["Adapter"] = pd.to_numeric(
    df["Adapter"],
    errors="coerce"
)


# ==========================================
# 10. CLEAN BATTERY LIFE
# ==========================================

# Examples:
# "Upto 12 Hrs Battery Life" -> 12
# "Upto 8 Hrs Battery Life"  -> 8
# "Upto 7.30 Hrs Battery Life" -> 7.30

df["Battery_Life"] = df["Battery_Life"].str.extract(
    r"([\d.]+)"
).astype(float)


# ==========================================
# 11. HANDLE MISSING VALUES
# ==========================================

# GPU has a few missing values.
# We don't want to delete those laptop records,
# so we replace missing GPU information with "Unknown".

df["GPU"] = df["GPU"].fillna("Unknown")

df["GPU_Brand"] = df["GPU_Brand"].fillna("Unknown")


# Battery life has many missing values.
# We use the median because it is more resistant
# to extreme values than the mean.

df["Battery_Life"] = df["Battery_Life"].fillna(
    df["Battery_Life"].median()
)


# Adapter also contains non-numeric values such as "no".
# Replace those missing values with the median.

df["Adapter"] = df["Adapter"].fillna(
    df["Adapter"].median()
)


# ==========================================
# 12. DISPLAY CLEANED DATA
# ==========================================

print("\n--------------------------------")
print("CLEANED DATA")
print("--------------------------------")

print(df.head())


print("\n--------------------------------")
print("DATA TYPES")
print("--------------------------------")

print(df.dtypes)


print("\n--------------------------------")
print("MISSING VALUES")
print("--------------------------------")

print(df.isnull().sum())


print("\n--------------------------------")
print("FINAL DATASET SHAPE")
print("--------------------------------")

print(df.shape)