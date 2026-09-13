"""
E-Commerce Customer Behavior Analysis
Step 2: Exploratory Data Analysis (EDA)
Author: Amit Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set up outputs directory
os.makedirs('../outputs', exist_ok=True)

# Set visual theme
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

print("Loading cleaned dataset...")
df = pd.read_csv('../data/ecommerce_cleaned.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

print("Starting Exploratory Data Analysis & generating plots...")

# 1. Age Distribution
plt.figure()
sns.histplot(df['age'], bins=20, kde=True, color='cyan')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('../outputs/01_age_distribution.png', bbox_inches='tight')
plt.close()

# 2. Total Amount Distribution
plt.figure()
sns.histplot(df['total_amount'], bins=30, kde=True, color='magenta')
plt.title('Transaction Value (Total Amount) Distribution')
plt.xlabel('Total Amount (INR)')
plt.ylabel('Frequency')
plt.savefig('../outputs/02_total_amount_distribution.png', bbox_inches='tight')
plt.close()

# 3. Category-wise Revenue Analysis
plt.figure()
cat_rev = df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.values, y=cat_rev.index, palette='viridis')
plt.title('Total Revenue by Product Category')
plt.xlabel('Revenue (INR)')
plt.ylabel('Category')
plt.savefig('../outputs/03_category_revenue.png', bbox_inches='tight')
plt.close()

# 4. Gender-based Purchasing Patterns (Quantity)
plt.figure()
sns.countplot(data=df, x='product_category', hue='gender', palette='Set2')
plt.title('Product Category Preferences by Gender')
plt.xticks(rotation=45)
plt.savefig('../outputs/04_category_by_gender.png', bbox_inches='tight')
plt.close()

# 5. Monthly Revenue Trends
plt.figure()
monthly_rev = df.groupby([df['transaction_date'].dt.to_period('M')])['total_amount'].sum()
monthly_rev.index = monthly_rev.index.astype(str)
sns.lineplot(x=monthly_rev.index, y=monthly_rev.values, marker='o', color='orange', linewidth=2)
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.ylabel('Total Revenue')
plt.xlabel('Month')
plt.savefig('../outputs/05_monthly_revenue_trend.png', bbox_inches='tight')
plt.close()

# 6. Payment Method Distribution
plt.figure()
payment_counts = df['payment_method'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Payment Method Distribution')
plt.savefig('../outputs/06_payment_methods.png', bbox_inches='tight')
plt.close()

# 7. Revenue by Membership Type
plt.figure()
sns.boxplot(data=df, x='membership_type', y='total_amount', order=['Bronze', 'Silver', 'Gold', 'Platinum'], palette='rocket')
plt.title('Transaction Size by Membership Tier')
plt.savefig('../outputs/07_membership_revenue.png', bbox_inches='tight')
plt.close()

# 8. Order Status Analysis (Return Rates)
plt.figure()
status_cat = pd.crosstab(df['product_category'], df['order_status'], normalize='index') * 100
status_cat.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(12,6))
plt.title('Order Status Proportion by Category (%)')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Percentage')
plt.savefig('../outputs/08_order_status_category.png', bbox_inches='tight')
plt.close()

# 9. Satisfaction Score Analysis
plt.figure()
sns.countplot(data=df, x='satisfaction_score', palette='autumn')
plt.title('Customer Satisfaction Score Distribution')
plt.savefig('../outputs/09_satisfaction_scores.png', bbox_inches='tight')
plt.close()

# 10. Correlation Heatmap
plt.figure(figsize=(10,8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('../outputs/10_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 11. Day of Week Purchasing Patterns
plt.figure()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df, x='order_day_of_week', order=day_order, palette='crest')
plt.title('Transactions by Day of Week')
plt.savefig('../outputs/11_day_of_week_txns.png', bbox_inches='tight')
plt.close()

# 12. City-wise Revenue
plt.figure()
city_rev = df.groupby('city')['total_amount'].sum().sort_values(ascending=False)
sns.barplot(x=city_rev.values, y=city_rev.index, palette='mako')
plt.title('Total Revenue by City')
plt.savefig('../outputs/12_city_revenue.png', bbox_inches='tight')
plt.close()

# 13. Discount Impact on Quantity
plt.figure()
sns.boxplot(data=df, x='discount_pct', y='quantity', palette='spring')
plt.title('Impact of Discount on Quantity Purchased')
plt.savefig('../outputs/13_discount_impact.png', bbox_inches='tight')
plt.close()

print("✅ EDA Complete! All 13 visualizations have been saved to the 'outputs' directory.")
