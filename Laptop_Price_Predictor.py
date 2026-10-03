import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET

df = pd.read_csv("laptop.csv")

# print("\nRaw HDD value for Lenovo ThinkBook:")
# print(df.loc[3655, ["Name", "SSD", "HDD"]])
# print("\nRaw SSD value containing 4098:")
# print(
#     df[df["SSD"].astype(str).str.contains("4098", na=False)]
#     [["Name", "SSD", "HDD"]]
# )

# print("Original Dataset Shape:", df.shape)

# 2. REMOVE UNNECESSARY COLUMN

df.drop("Unnamed: 0", axis=1, inplace=True)

# 3. CLEAN RAM

df["RAM"] = df["RAM"].str.extract(r"(\d+)").astype(float)

# 4. CLEAN PROCESSOR SPEED

df["Ghz"] = df["Ghz"].str.extract(r"([\d.]+)").astype(float)

# 5. CLEAN PROCESSOR BRAND

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

# print(df["Processor_Brand"].value_counts())

# 6. CLEAN DISPLAY SIZE

df["Display"] = df["Display"].str.extract(r"([\d.]+)").astype(float)
df["Display"] = df["Display"].fillna(df["Display"].median())

# 7. FUNCTION TO CONVERT STORAGE

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

# 8. CLEAN SSD

df["SSD"] = df["SSD"].apply(convert_storage)

# 9. CLEAN HDD

df["HDD"] = df["HDD"].apply(convert_storage)

# 10. CLEAN ADAPTER

df["Adapter"] = pd.to_numeric(
    df["Adapter"],
    errors="coerce"
)

# 11. CLEAN BATTERY LIFE

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

# 12. HANDLE MISSING VALUES

df["GPU"] = df["GPU"].fillna("Unknown")
df["GPU_Brand"] = df["GPU_Brand"].fillna("Unknown")

df["Adapter"] = df["Adapter"].fillna(
    df["Adapter"].median()
)

# print("\nBattery Life Statistics:")
# print(df["Battery_Life"].describe())

# print("\nLargest Battery Life Values:")
# print(df["Battery_Life"].sort_values(ascending=False).head(20))

# 13. DISPLAY CLEANED DATA

# print("\n--------------------------------")
# print("CLEANED DATA")
# print("--------------------------------")
# print(df.head())

# print("\n--------------------------------")
# print("DATA TYPES")
# print("--------------------------------")
# print(df.dtypes)

# print("\n--------------------------------")
# print("MISSING VALUES")
# print("--------------------------------")
# print(df.isnull().sum())

# print("\n--------------------------------")
# print("FINAL DATASET SHAPE")
# print("--------------------------------")
# print(df.shape)

# 14. EDA — BATTERY LIFE VS PRICE

# plt.figure(figsize=(10, 6))

# sns.scatterplot(
#     data=df,
#     x="Battery_Life",
#     y="Price",
#     alpha=0.5
# )

# plt.title("Battery Life vs Laptop Price")
# plt.xlabel("Battery Life (Hours)")
# plt.ylabel("Price")

# plt.show()

# 15. EDA — PRICE DISTRIBUTION

# plt.figure(figsize=(10, 6))
# sns.histplot(df["Price"], bins=30, kde=True)
# plt.title("Laptop Price Distribution")
# plt.xlabel("Price")
# plt.ylabel("Number of Laptops")
# plt.show()

# 16. EDA — GPU VS PRICE

# plt.figure(figsize=(14, 7))

# gpu_order = df.groupby("GPU")["Price"].median().sort_values(
#     ascending=False
# ).head(15).index

# sns.boxplot(
#     data=df[df["GPU"].isin(gpu_order)],
#     x="GPU",
#     y="Price"
# )

# plt.title("GPU vs Laptop Price")
# plt.xlabel("GPU")
# plt.ylabel("Price")
# plt.xticks(rotation=45)
# plt.show()

# 17. EDA — PROCESSOR BRAND VS PRICE

# plt.figure(figsize=(10, 6))

# sns.boxplot(
#     data=df,
#     x="Processor_Brand",
#     y="Price"
# )

# plt.title("Processor Brand vs Laptop Price")
# plt.xlabel("Processor Brand")
# plt.ylabel("Price")
# plt.show()

# 18. EDA — PROCESSOR MODEL VS PRICE

# plt.figure(figsize=(14, 7))

# processor_order = df["Processor_Name"].value_counts().head(15).index

# sns.boxplot(
#     data=df[df["Processor_Name"].isin(processor_order)],
#     x="Processor_Name",
#     y="Price"
# )

# plt.title("Processor Model vs Laptop Price")
# plt.xlabel("Processor")
# plt.ylabel("Price")
# plt.xticks(rotation=45)
# plt.show()

# 19. EDA — DISPLAY SIZE VS PRICE

# plt.figure(figsize=(10, 6))

# sns.scatterplot(
#     data=df,
#     x="Display",
#     y="Price",
#     alpha=0.5
# )

# plt.title("Display Size vs Laptop Price")
# plt.xlabel("Display Size (inches)")
# plt.ylabel("Price")
# plt.show()

# 20. EDA — BATTERY LIFE VS PRICE

# plt.figure(figsize=(10, 6))

# sns.scatterplot(
#     data=df,
#     x="Battery_Life",
#     y="Price",
#     alpha=0.5
# )

# plt.title("Battery Life vs Laptop Price")
# plt.xlabel("Battery Life (Hours)")
# plt.ylabel("Price")
# plt.show()

# 21. CORRELATION ANALYSIS

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

# print("\nCorrelation with Price:")
# print(correlation["Price"].sort_values(ascending=False))

# 22. CORRELATION HEATMAP

# plt.figure(figsize=(10, 7))

# sns.heatmap(
#     correlation,
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f"
# )

# plt.title("Correlation Heatmap")
# plt.show()

# 23. STORAGE ANALYSIS

# print("\nOriginal SSD values:")
# print(df["SSD"].value_counts().sort_index())

# print("\nOriginal HDD values:")
# print(df["HDD"].value_counts().sort_index())

# print(df[df["HDD"] == 1][["Name", "HDD"]])

# print(df["SSD"].value_counts().sort_index())

# 24. PRICE OUTLIER ANALYSIS

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_outliers = df[
    (df["Price"] < lower_bound) |
    (df["Price"] > upper_bound)
]

# print("\nPrice Outlier Analysis:")
# print("Q1:", Q1)
# print("Q3:", Q3)
# print("IQR:", IQR)
# print("Lower Bound:", lower_bound)
# print("Upper Bound:", upper_bound)
# print("Number of Outliers:", len(price_outliers))

# 25. MOST EXPENSIVE LAPTOPS

# print("\nMost Expensive Laptops:")
# print(
#     df.nlargest(
#         10,
#         "Price"
#     )[["Name", "Brand", "Price", "RAM", "SSD", "GPU"]]
# )

# 26. PRICE OUTLIER BOXPLOT

# plt.figure(figsize=(10, 6))

# sns.boxplot(
#     y=df["Price"]
# )

# plt.title("Laptop Price Outlier Analysis")
# plt.ylabel("Price")

# plt.show()

# 27. SEPARATE FEATURES AND TARGET

X = df.drop("Price", axis=1)
y = df["Price"]

# print("\nFeatures:")
# print(X.columns.tolist())

# print("\nTarget:")
# print(y.name)

# 28. REMOVE HIGH-CARDINALITY FEATURE

X = X.drop(["Name"], axis=1)

# print("Remaining Features:")
# print(X.columns.tolist())

# 29. SEPARATE NUMERICAL AND CATEGORICAL FEATURES

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()

# print("Numerical Features:")
# print(numerical_features)

# print("\nCategorical Features:")
# print(categorical_features)

# --- Split Train and Test Sets ---

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --- Preprocessing Pipelines ---

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --- Model Training ---

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)

# --- Evaluation Metrics ---

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#print("Mean Absolute Error:", mae)
#print("Root Mean Squared Error:", rmse)
#print("R² Score:", r2)

# ---train linear regression model

from sklearn.linear_model import LinearRegression

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# --- Evaluation Metrics for Linear Regression ---

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

#print("Linear Regression")
#print("Mean Absolute Error:", linear_mae)
#print("Root Mean Squared Error:", linear_rmse)
#print("R² Score:", linear_r2)

# --- Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

gradient_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

gradient_model.fit(X_train, y_train)
gradient_pred = gradient_model.predict(X_test)

# --- Evaluation Metrics for Gradient Boosting Regressor ---

gradient_mae = mean_absolute_error(y_test, gradient_pred)
gradient_rmse = np.sqrt(mean_squared_error(y_test, gradient_pred))
gradient_r2 = r2_score(y_test, gradient_pred)

#print("Gradient Boosting")
#print("Mean Absolute Error:", gradient_mae)
#print("Root Mean Squared Error:", gradient_rmse)
#print("R² Score:", gradient_r2)

#Random Forest with more trees

tuned_rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=500,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

tuned_rf_model.fit(X_train, y_train)

tuned_rf_pred = tuned_rf_model.predict(X_test)

# --- Evaluation Metrics for Tuned Random Forest ---

tuned_rf_mae = mean_absolute_error(y_test, tuned_rf_pred)
tuned_rf_rmse = np.sqrt(mean_squared_error(y_test, tuned_rf_pred))
tuned_rf_r2 = r2_score(y_test, tuned_rf_pred)

#print("Tuned Random Forest")
#print("Mean Absolute Error:", tuned_rf_mae)
#print("Root Mean Squared Error:", tuned_rf_rmse)
#print("R² Score:", tuned_rf_r2)

# --- Extra Trees Regressor

from sklearn.ensemble import ExtraTreesRegressor

extra_trees_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

extra_trees_model.fit(X_train, y_train)

extra_trees_pred = extra_trees_model.predict(X_test)

# --- Evaluation Metrics for Extra Trees Regressor ---

extra_trees_mae = mean_absolute_error(y_test, extra_trees_pred)
extra_trees_rmse = np.sqrt(mean_squared_error(y_test, extra_trees_pred))
extra_trees_r2 = r2_score(y_test, extra_trees_pred)

#print("Extra Trees")
#print("Mean Absolute Error:", extra_trees_mae)
#print("Root Mean Squared Error:", extra_trees_rmse)
#print("R² Score:", extra_trees_r2)

## --- Model Comparison ---

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Linear Regression",
        "Gradient Boosting",
        "Tuned Random Forest",
        "Extra Trees"
    ],
    "MAE": [
        mae,
        linear_mae,
        gradient_mae,
        tuned_rf_mae,
        extra_trees_mae
    ],
    "RMSE": [
        rmse,
        linear_rmse,
        gradient_rmse,
        tuned_rf_rmse,
        extra_trees_rmse
    ],
    "R2 Score": [
        r2,
        linear_r2,
        gradient_r2,
        tuned_rf_r2,
        extra_trees_r2
    ]
})

print("\nModel Comparison:")
print(results.sort_values("R2 Score", ascending=False).to_string(index=False))