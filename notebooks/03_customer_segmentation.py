"""
E-Commerce Customer Behavior Analysis
Step 3: Customer Segmentation & Cohort Analysis
Author: Amit Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="deep")

print("Loading data for Segmentation...")
df = pd.read_csv('../data/ecommerce_cleaned.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# ---- 1. RFM Analysis ----
print("\n" + "="*50)
print("PERFORMING RFM ANALYSIS")
print("="*50)

# Assume current date is 1 day after the max transaction date in the dataset
current_date = df['transaction_date'].max() + dt.timedelta(days=1)

rfm = df.groupby('customer_id').agg({
    'transaction_date': lambda x: (current_date - x.max()).days,
    'transaction_id': 'count',
    'total_amount': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# Assign scores 1-4
r_labels = range(4, 0, -1) # 4 is best (lowest recency)
f_labels = range(1, 5)     # 4 is best (highest frequency)
m_labels = range(1, 5)     # 4 is best (highest monetary)

# Calculate quantiles, drop duplicates if edges are same
r_bins = pd.qcut(rfm['Recency'], q=4, labels=r_labels, duplicates='drop')
f_bins = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_bins = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)

rfm['R'] = r_bins
rfm['F'] = f_bins
rfm['M'] = m_bins

rfm['RFM_Score'] = rfm[['R', 'F', 'M']].astype(str).sum(axis=1)

# Segment function
def segment_customer(df):
    if df['RFM_Score'] in ['444', '443', '434', '344']:
        return 'Champions'
    elif df['R'] >= 3 and df['F'] >= 3:
        return 'Loyal Customers'
    elif df['R'] >= 3 and df['F'] <= 2:
        return 'Recent & Low Freq'
    elif df['R'] <= 2 and df['F'] >= 3:
        return 'At Risk'
    elif df['R'] <= 2 and df['F'] <= 2:
        return 'Lost/Hibernating'
    return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("\nRFM Segments Distribution:")
print(rfm['Segment'].value_counts())

plt.figure(figsize=(10,6))
sns.countplot(data=rfm, y='Segment', order=rfm['Segment'].value_counts().index, palette='magma')
plt.title('Customer Segments based on RFM Analysis')
plt.xlabel('Number of Customers')
plt.savefig('../outputs/14_rfm_segments.png', bbox_inches='tight')
plt.close()

# ---- 2. Cohort Analysis ----
print("\n" + "="*50)
print("PERFORMING COHORT ANALYSIS")
print("="*50)

df['cohort_month'] = df.groupby('customer_id')['transaction_date'].transform('min').dt.to_period('M')
df['transaction_month'] = df['transaction_date'].dt.to_period('M')

cohort_data = df.groupby(['cohort_month', 'transaction_month'])['customer_id'].nunique().reset_index()

# Calculate cohort index (months since first purchase)
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

cohort_year, cohort_month = get_date_int(cohort_data, 'cohort_month')
transaction_year, transaction_month = get_date_int(cohort_data, 'transaction_month')

years_diff = transaction_year - cohort_year
months_diff = transaction_month - cohort_month
cohort_data['cohort_index'] = years_diff * 12 + months_diff + 1

cohort_counts = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')
cohort_sizes = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_sizes, axis=0) * 100

plt.figure(figsize=(16, 10))
sns.heatmap(retention, annot=True, fmt='.1f', cmap='YlGnBu', vmin=0, vmax=100)
plt.title('Customer Retention Heatmap (%)')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.savefig('../outputs/15_cohort_retention.png', bbox_inches='tight')
plt.close()

# ---- 3. Churn Analysis ----
print("\n" + "="*50)
print("CHURN ANALYSIS")
print("="*50)

# Define churn: no purchase in last 90 days
rfm['Churned'] = rfm['Recency'] > 90
churn_rate = rfm['Churned'].mean() * 100
print(f"Overall Churn Rate: {churn_rate:.2f}% (Inactive for > 90 days)")

plt.figure(figsize=(8,8))
plt.pie([rfm['Churned'].sum(), (~rfm['Churned']).sum()], labels=['Churned (>90 days)', 'Active'], 
        autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title('Customer Churn Proportion')
plt.savefig('../outputs/16_churn_proportion.png', bbox_inches='tight')
plt.close()

print("\n✅ Customer Segmentation & Cohort Analysis Complete!")
