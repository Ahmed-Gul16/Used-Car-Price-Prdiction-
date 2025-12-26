Pakistani Used Car Price Prediction

This project implements a machine learning pipeline to predict the market value of used cars in Pakistan with 88% accuracy. It utilizes a Random Forest Regressor and specialized feature engineering to handle the high volatility of the local automotive market.

Project Overview

The Pakistani used car market is uniquely affected by currency devaluation, "on-money" premiums, and registration-based pricing. This project quantifies these factors to provide an objective valuation tool.

Key Features

Data Cleaning: Handled extreme outliers (engine CC typos) using Group-wise Median Imputation.

EV Mapping: Engineered a conversion logic for Electric Vehicle battery capacity (kWh) to performance-equivalent CC.

Inflation Modeling: Captured the vertical price spikes post-2018 caused by local economic shifts.

Geographical Analysis: Modeled the price premium associated with Islamabad and Lahore registrations.

Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib

Algorithm: Random Forest Regressor (Optimized Max Depth: 10)

Results

R² Score: 0.8805

Mean Absolute Error (MAE): ±563,000 PKR

Insights: Confirmed that 'Year' and 'Engine' are the dominant price predictors in the current economy.

Repository Structure

car_price_prediction.ipynb: Full data cleaning, EDA, and modeling pipeline.

Report.pdf: Comprehensive project report with business insights.

final_car_price_model.pkl: (Optional) Saved model components for deployment.

Acknowledgments

Special thanks to our lab instructor for the foundational Python guidance that made this independent project possible.
