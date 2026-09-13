"""
E-Commerce Customer Behavior Analysis
Step 1: Data Cleaning & Preprocessing
Author: Amit Kumar
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

# Ensure data output path exists
os.makedirs('../data', exist_ok=True)

print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)
try:
    df = pd.read_csv('../data/ecommerce_transactions.csv')
    print(f"Successfully loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print("ERROR: Data file not found. Please run 00_generate_data.py first.")
    exit(1)

# ---- Data Overview ----
print("\n" + "=" * 60)
print("2. INITIAL DATA OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumn Types:\n{df.dtypes}")
print(f"\nMissing Values Before Cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print(f"Duplicate Transaction IDs: {df['transaction_id'].duplicated().sum()}")

# ---- Handle Duplicates ----
print("\n" + "=" * 60)
print("3. HANDLING DUPLICATES")
print("=" * 60)
df = df.drop_duplicates(subset='transaction_id', keep='first')
print(f"Dataset shape after removing duplicate transaction IDs: {df.shape}")

# ---- Handle Missing Values ----
print("\n" + "=" * 60)
print("4. HANDLING MISSING VALUES")
print("=" * 60)
# Fill satisfaction_score with median
med_score = df['satisfaction_score'].median()
df['satisfaction_score'] = df['satisfaction_score'].fillna(med_score)
print(f"Filled missing satisfaction_score with overall median: {med_score}")

# Fill days_since_last_purchase with median per customer
df['days_since_last_purchase'] = df.groupby('customer_id')['days_since_last_purchase'].transform(
    lambda x: x.fillna(x.median())
)
# For customers with completely missing days_since_last_purchase, fill with global median
global_days_med = df['days_since_last_purchase'].median()
df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(global_days_med)
print("Filled missing days_since_last_purchase using customer-specific medians, then global median.")

# ---- Data Type Conversion ----
print("\n" + "=" * 60)
print("5. DATA TYPE CONVERSIONS")
print("=" * 60)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
print("Converted 'signup_date' and 'transaction_date' to datetime format.")

# ---- Feature Engineering ----
print("\n" + "=" * 60)
print("6. FEATURE ENGINEERING")
print("=" * 60)
df['total_amount'] = df['quantity'] * df['unit_price'] * (1 - df['discount_pct']/100)
df['order_month'] = df['transaction_date'].dt.month
df['order_year'] = df['transaction_date'].dt.year
df['order_quarter'] = df['transaction_date'].dt.quarter
df['order_day_of_week'] = df['transaction_date'].dt.day_name()
df['customer_tenure_days'] = (df['transaction_date'] - df['signup_date']).dt.days

print("Created new features:")
print("- total_amount (quantity * price after discount)")
print("- order_month, order_year, order_quarter, order_day_of_week")
print("- customer_tenure_days (days between signup and transaction)")

# ---- Outlier Detection & Treatment ----
print("\n" + "=" * 60)
print("7. OUTLIER DETECTION (total_amount)")
print("=" * 60)
Q1 = df['total_amount'].quantile(0.25)
Q3 = df['total_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

outliers = df[(df['total_amount'] < lower_bound) | (df['total_amount'] > upper_bound)]
print(f"Outliers detected: {len(outliers)}")
print(f"Capping values between {lower_bound:.2f} and {upper_bound:.2f}")

# Cap outliers instead of removing
df['total_amount'] = df['total_amount'].clip(lower=lower_bound, upper=upper_bound)

# ---- Final Checks and Save ----
print("\n" + "=" * 60)
print("8. FINAL DATASET EXPORT")
print("=" * 60)
df.to_csv('../data/ecommerce_cleaned.csv', index=False)
print(f"Cleaned data saved to '../data/ecommerce_cleaned.csv'")
print(f"Final dataset shape: {df.shape}")
print(f"Final Missing Values:\n{df.isnull().sum().sum()}")
print("\n✅ Data Cleaning Complete!")
