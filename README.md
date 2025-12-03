Mortality Trends Analysis
Advanced Data Analysis | EDA | Time-Series Modeling | Forecasting

This project analyzes weekly mortality data from the United States using advanced data-science techniques including exploratory data analysis (EDA), feature engineering, trend decomposition, and machine-learning-based forecasting.

The goal is to identify major mortality patterns, understand cause-wise trends, and build predictive models to forecast all-cause mortality.

 Project Highlights

 Performed comprehensive EDA to understand mortality behavior

 Built feature-engineered time-series dataset

 Applied linear regression forecasting using lag & moving-average features

 Conducted trend + seasonality decomposition

 Created heatmaps, time-series plots, and stacked area charts

 Extracted meaningful insights such as peak mortality weeks and correlations

 Dataset

Source: Weekly mortality records (Excel format) containing:

All-Cause deaths

COVID-19 deaths

Major causes (heart disease, cancer, etc.)

Date of week ending

Jurisdiction (state names)

 Technologies Used

Python 3

Pandas, NumPy

Matplotlib, Seaborn

Statsmodels (seasonal decomposition)

Scikit-learn (machine learning)

 Project Workflow
1. Import & Clean Data

Loaded Excel dataset

Converted date fields

Filtered U.S-level aggregation

Filled missing values

Sorted chronologically

2. Feature Engineering

Engineered several predictive & analytical features:

Lag variables (1-week lag)

Moving Averages (4-week average)

Week number

Year

Rolling trend signals

3. Exploratory Data Analysis

Includes:

Time-series trend plots

Correlation matrix

Distribution of causes

Stacked cause-wise area charts

Jurisdiction heatmap

4. Time-Series Trend Decomposition

Using seasonal_decompose():

Trend

Seasonal patterns

Residual noise

5. Machine Learning Forecast

Built a Linear Regression model using:

AllCause_Lag1

COVID_Lag1

AllCause_MA4

COVID_MA4

Evaluated using MAE & R², with a prediction visualization.

 6. Results & Key Insights
 Peak Mortality

Highest COVID-19 mortality week identified

Highest All-Cause mortality week identified

 Correlation

COVID-19 and All-Cause Mortality correlation example:

Strong positive correlation → pandemic impact visible in overall mortality.

 Year-wise mortality totals

Helps track increase or decrease over years.

Visualizations Included

All-Cause vs Natural Deaths Trend

Correlation Heatmap

Time-Series Decomposition

Prediction vs Actual Line Plot

Stacked Area Chart for top causes

Jurisdiction Heatmap 
