Sales Forecasting Project – Walmart Weekly Sales Prediction

Description:
This project focuses on predicting weekly sales for Walmart stores using historical sales data. The goal is to provide accurate forecasts that can help businesses plan inventory, staffing, and promotions. The project demonstrates the full pipeline of a regression-based time series forecasting workflow, including data preprocessing, feature engineering, model training, evaluation, and visualization.

Dataset:
The dataset used is the Walmart Sales Forecast dataset from Kaggle, containing historical sales, store information, and economic indicators.

Key Features:

Store ID

Date (Year, Month, Week, DayOfWeek)

Holiday flag

IsWeekend indicator

Economic factors: Fuel Price, CPI, Unemployment

Technologies & Libraries:

Python

Pandas, NumPy

Scikit-learn

LightGBM

Matplotlib, Seaborn

Project Workflow:

Data Preprocessing – Clean data, handle missing values, create new features.

Feature Engineering – Extract time-based features (Year, Month, Week, DayOfWeek, IsWeekend).

Model Training – Train a LightGBM regression model and save it for predictions.

Prediction – Load new data, apply preprocessing, and generate forecasts.

Evaluation & Visualization – Compute MAE, RMSE, MAPE and plot actual vs predicted sales.

Metrics:

MAE: 309,120

RMSE: 413,219

MAPE: 40.91%

Outputs:

predictions.csv – Predicted weekly sales

actual_vs_predicted_recent.png – Visualization of actual vs predicted sales for recent weeks
