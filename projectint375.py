import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading dataset
df = pd.read_excel("python dataset.xlsx")
df.head()

#data cleaning

#handling missing values
df.fillna(0, inplace=True)
plt.show()

#convert dataTpyes
df['Week Ending Date'] = pd.to_datetime(df['Week Ending Date'])

#Exploratory Data Analysis (EDA)
df.describe()

#Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Causes of Death")
plt.show()

#Total Deaths Over Time
df_grouped = df.groupby("Week Ending Date")[["All Cause", "Natural Cause"]].sum()

df_grouped.plot(figsize=(12,6))
plt.title("Trend of All-Cause and Natural Deaths Over Time")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.grid(True)
plt.show()

# Visualizations with Seaborn & Matplotlib
# COVID vs Heart Disease
df_plot = df.groupby("Week Ending Date")[
    ["COVID-19 (U071, Underlying Cause of Death)", "Diseases of heart (I00-I09,I11,I13,I20-I51)"]
].sum()

df_plot.plot(figsize=(12,6))
plt.title("COVID-19 vs Heart Disease Deaths Over Time")
plt.ylabel("Number of Deaths")
plt.show()

#Stacked Area Chart — Trend Comparison Over Time
df_grouped = df.groupby("Week Ending Date")[
    ['COVID-19 (U071, Underlying Cause of Death)', 
     'Diseases of heart (I00-I09,I11,I13,I20-I51)', 
     'Malignant neoplasms (C00-C97)']
].sum()

df_grouped.plot.area(figsize=(12, 6), colormap='Set2')
plt.title("Stacked Area Chart of Major Causes of Death Over Time")
plt.xlabel("Week Ending Date")
plt.ylabel("Death Count")
plt.show()

#Heatmap — Weekly Trend of COVID-19 Deaths (Pivoted Table)
heatmap_df = df.pivot_table(index='Jurisdiction of Occurrence', 
                            columns='Week Ending Date', 
                            values='COVID-19 (U071, Underlying Cause of Death)', 
                            aggfunc='sum')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_df.fillna(0), cmap='YlOrRd')
plt.title("Heatmap of COVID-19 Deaths per Jurisdiction Over Time")
plt.show()

#Pie Chart — Proportion of Selected Causes of Death
selected = df[[
    'COVID-19 (U071, Underlying Cause of Death)', 
    'Diseases of heart (I00-I09,I11,I13,I20-I51)', 
    'Malignant neoplasms (C00-C97)', 
    'Alzheimer disease (G30)'
]].sum()

plt.figure(figsize=(7, 7))
plt.pie(selected, labels=selected.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
plt.title("Proportion of Selected Causes of Death")
plt.show()

#Line Plot — Comparing Weekly Trends of Diabetes & Alzheimer’s
line_data = df.groupby("Week Ending Date")[
    ["Diabetes mellitus (E10-E14)", "Alzheimer disease (G30)"]
].sum()

line_data.plot(figsize=(12,6), color=["teal", "orchid"])
plt.title("Weekly Deaths: Diabetes vs Alzheimer’s")
plt.ylabel("Number of Deaths")
plt.grid(True)
plt.show()

# Top 5 Causes of Death (Total)
top_causes = df[[
    'COVID-19 (U071, Underlying Cause of Death)',
    'Diseases of heart (I00-I09,I11,I13,I20-I51)',
    'Malignant neoplasms (C00-C97)',
    'Diabetes mellitus (E10-E14)',
    'Alzheimer disease (G30)'
]].sum().sort_values(ascending=False)

top_causes.plot(kind='barh', color='skyblue')
plt.title("Top 5 Causes of Death (Total)")
plt.xlabel("Total Deaths")
plt.show()

#Insight Extraction (EDA)
#Use Pandas & summary plots to find:
#Trends: Did COVID-19 deaths follow a wave pattern?
#Region Analysis: Which jurisdictions were most affected?
#Correlation: Are certain diseases correlated?
top_states = df.groupby("Jurisdiction of Occurrence")["All Cause"].sum().sort_values(ascending=False).head(10)

top_states.plot(kind='bar', color='coral')
plt.title("Top 10 States by All-Cause Deaths")
plt.ylabel("Deaths")
plt.show()

#Conclusion
#Wrap up with findings:

#"COVID-19 had sharp spikes in 2020 and 2021"

#"Heart disease continues to be a major cause"

#"High mortality jurisdictions include X, Y, Z"
print("COVID-19 had sharp spikes in 2020 and 2021")
print("Heart disease continues to be a major cause")
print("Top affected jurisdictions include: ", top_states.index.tolist())









