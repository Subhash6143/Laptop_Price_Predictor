import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET

df = pd.read_csv("laptop.csv")

print("\nRaw HDD value for Lenovo ThinkBook:")
print(df.loc[3655, ["Name", "SSD", "HDD"]])
print("\nRaw SSD value containing 4098:")
print(
    df[df["SSD"].astype(str).str.contains("4098", na=False)]
    [["Name", "SSD", "HDD"]]
)

print("Original Dataset Shape:", df.shape)

# 2. REMOVE UNNECESSARY COLUMN

df.drop("Unnamed: 0", axis=1, inplace=True)

# 3. CLEAN RAM

df["RAM"] = df["RAM"].str.extract(r"(\d+)").astype(float)

# 4. CLEAN PROCESSOR SPEED

df["Ghz"] = df["Ghz"].str.extract(r"([\d.]+)").astype(float)

valid_processor_brands = [
    "Intel",
    "AMD",
    "Apple",
    "MediaTek",
    "Qualcomm",
    "Microsoft"
]

df.loc[
    ~df["Processor_Brand"].isin(valid_processor_brands),
    "Processor_Brand"
] = "Unknown"

print(df["Processor_Brand"].value_counts())

# 5. CLEAN DISPLAY SIZE

df["Display"] = df["Display"].str.extract(r"([\d.]+)").astype(float)
df["Display"] = df["Display"].fillna(df["Display"].median())

# 6. FUNCTION TO CONVERT STORAGE

def convert_storage(value):

    value = str(value).upper().strip()

    number = pd.to_numeric(
        pd.Series(value).str.extract(r"([\d.]+)")[0].iloc[0],
        errors="coerce"
    )

    if pd.isna(number):
        return 0

    if "TB" in value:
        number = number * 1024

    return number

# 7. CLEAN SSD

df["SSD"] = df["SSD"].apply(convert_storage)

# 8. CLEAN HDD

df["HDD"] = df["HDD"].apply(convert_storage)

# 9. CLEAN ADAPTER

df["Adapter"] = pd.to_numeric(
    df["Adapter"],
    errors="coerce"
)

# 10. CLEAN BATTERY LIFE

df["Battery_Life"] = df["Battery_Life"].where(
    ~df["Battery_Life"].str.contains("Adapter", case=False, na=False),
    np.nan
)

df["Battery_Life"] = df["Battery_Life"].str.extract(
    r"([\d.]+)"
).astype(float)

df["Battery_Life"] = df["Battery_Life"].fillna(

    df["Battery_Life"].median()
)

# 11. HANDLE MISSING VALUES

df["GPU"] = df["GPU"].fillna("Unknown")
df["GPU_Brand"] = df["GPU_Brand"].fillna("Unknown")


df["Adapter"] = df["Adapter"].fillna(
    df["Adapter"].median()
)

print("\nBattery Life Statistics:")
print(df["Battery_Life"].describe())

print("\nLargest Battery Life Values:")
print(df["Battery_Life"].sort_values(ascending=False).head(20))

# 12. DISPLAY CLEANED DATA

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

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Battery_Life",
    y="Price",
    alpha=0.5
)

plt.title("Battery Life vs Laptop Price")
plt.xlabel("Battery Life (Hours)")
plt.ylabel("Price")

plt.show()

"""
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Laptop Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Laptops")
plt.show()



plt.figure(figsize=(14, 7))
gpu_order = df.groupby("GPU")["Price"].median().sort_values(ascending=False).head(15).index
sns.boxplot(
    data=df[df["GPU"].isin(gpu_order)],
    x="GPU",
    y="Price"
)
plt.title("GPU vs Laptop Price")
plt.xlabel("GPU")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Processor_Brand",
    y="Price"
)
plt.title("Processor Brand vs Laptop Price")
plt.xlabel("Processor Brand")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(14, 7))
processor_order = df["Processor_Name"].value_counts().head(15).index
sns.boxplot(
    data=df[df["Processor_Name"].isin(processor_order)],
    x="Processor_Name",
    y="Price"
)
plt.title("Processor Model vs Laptop Price")
plt.xlabel("Processor")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Display",
    y="Price",
    alpha=0.5
)
plt.title("Display Size vs Laptop Price")
plt.xlabel("Display Size (inches)")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Battery_Life",
    y="Price",
    alpha=0.5
)
plt.title("Battery Life vs Laptop Price")
plt.xlabel("Battery Life (Hours)")
plt.ylabel("Price")
plt.show()


"""

numeric_columns = [
    "Price",
    "RAM",
    "Ghz",
    "Display",
    "SSD",
    "HDD",
    "Adapter",
    "Battery_Life"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation with Price:")
print(correlation["Price"].sort_values(ascending=False))

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

print("\nOriginal SSD values:")
print(df["SSD"].value_counts().sort_index())

print("\nOriginal HDD values:")
print(df["HDD"].value_counts().sort_index())
print(df[df["HDD"] == 1][["Name", "HDD"]])
print(df["SSD"].value_counts().sort_index())

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_outliers = df[
    (df["Price"] < lower_bound) |
    (df["Price"] > upper_bound)
]

print("\nPrice Outlier Analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Outliers:", len(price_outliers))

print("\nMost Expensive Laptops:")
print(
    df.nlargest(
        10,
        "Price"
    )[["Name", "Brand", "Price", "RAM", "SSD", "GPU"]]
)


plt.figure(figsize=(10, 6))

sns.boxplot(
    y=df["Price"]
)
plt.title("Laptop Price Outlier Analysis")
plt.ylabel("Price")

plt.show()