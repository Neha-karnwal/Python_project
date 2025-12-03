# ----------------------------
# Mortality Trends Analysis Project
# Data Analysis | EDA | Time-Series ML | Forecasting
# ----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_excel("python dataset.xlsx")

# Standard column cleanup
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------
# 2. Data Cleaning
# --------------------------------------------

# Convert date column
df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"])

# Replace missing values
df.fillna(0, inplace=True)

# Keep only USA-level records (if needed)
df = df[df["Jurisdiction of Occurrence"] == "United States"]

# Sort by time
df = df.sort_values("Week Ending Date")

# --------------------------------------------
# 3. Feature Engineering
# --------------------------------------------

df["year"] = df["Week Ending Date"].dt.year
df["week"] = df["Week Ending Date"].dt.isocalendar().week

# Lag features for ML models
df["AllCause_Lag1"] = df["All Cause"].shift(1)
df["COVID_Lag1"] = df["COVID-19"].shift(1)

# 4-week Rolling Averages
df["AllCause_MA4"] = df["All Cause"].rolling(4).mean()
df["COVID_MA4"] = df["COVID-19"].rolling(4).mean()

# Drop initial NaNs from rolling windows
df = df.dropna()

# --------------------------------------------
# 4. Exploratory Data Analysis
# --------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Week Ending Date"], df["All Cause"], label="All Cause Deaths")
plt.plot(df["Week Ending Date"], df["Natural Cause"], label="Natural Deaths")
plt.title("Mortality Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.legend()
plt.grid(True)
plt.show()

# Correlation Matrix
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix of Mortality Features")
plt.show()

# --------------------------------------------
# 5. Time-Series Decomposition (Trend + Seasonality)
# --------------------------------------------

ts = df.set_index("Week Ending Date")["All Cause"]
result = seasonal_decompose(ts, model="additive", period=52)

result.plot()
plt.suptitle("Time-Series Decomposition: All-Cause Mortality", y=1.02)
plt.show()

# --------------------------------------------
# 6. ML Model → Predicting Mortality (Simple Regression)
# --------------------------------------------

features = ["AllCause_Lag1", "COVID_Lag1", "AllCause_MA4", "COVID_MA4"]
X = df[features]
y = df["All Cause"]

# Train-test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, pred, label="Predicted")
plt.title("All-Cause Mortality Prediction")
plt.legend()
plt.show()

# --------------------------------------------
# 7. Stacked Area Plot of Major Causes
# --------------------------------------------
major_causes = df.set_index("Week Ending Date")[[
    "COVID-19",
    "Diseases of heart",
    "Malignant neoplasms",
]]

major_causes.plot.area(figsize=(12, 6), colormap="Set2")
plt.title("Major Causes of Death — Trend Comparison")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.show()

# --------------------------------------------
# 8. Heatmap by State (If multiple states exist)
# --------------------------------------------
heatmap_df = df.pivot_table(
    index="Jurisdiction of Occurrence",
    columns="Week Ending Date",
    values="COVID-19",
    aggfunc="sum"
)

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_df, cmap="YlOrRd")
plt.title("COVID-19 Mortality Heatmap Across Jurisdictions")
plt.show()

# --------------------------------------------
# 9. Insight Extraction
# --------------------------------------------
peak_covid_week = df.loc[df["COVID-19"].idxmax()]["Week Ending Date"]
peak_allcause_week = df.loc[df["All Cause"].idxmax()]["Week Ending Date"]

print("\n----- INSIGHTS -----")
print(f"Highest COVID-19 deaths recorded during week: {peak_covid_week.date()}")
print(f"Highest overall mortality recorded on: {peak_allcause_week.date()}")

corr = df["COVID-19"].corr(df["All Cause"])
print(f"Correlation between COVID-19 & All-Cause Mortality: {corr:.2f}")

yearly_summary = df.groupby("year")["All Cause"].sum()
print("\nYearly Mortality Totals:")
print(yearly_summary)
