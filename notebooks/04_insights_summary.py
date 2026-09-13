"""
E-Commerce Customer Behavior Analysis
Step 4: Insights & Summary Dashboard
Author: Amit Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../outputs', exist_ok=True)
plt.style.use('dark_background')

print("=" * 60)
print("PROJECT SUMMARY & FINAL DASHBOARD")
print("=" * 60)

# Load data
df = pd.read_csv('../data/ecommerce_cleaned.csv')

total_revenue = df['total_amount'].sum()
total_orders = df['transaction_id'].nunique()
avg_order_val = df['total_amount'].mean()
total_customers = df['customer_id'].nunique()

print(f"\n--- HIGH-LEVEL METRICS ---")
print(f"Total Revenue:      ₹{total_revenue:,.2f}")
print(f"Total Orders:       {total_orders:,}")
print(f"Total Customers:    {total_customers:,}")
print(f"Avg Order Value:    ₹{avg_order_val:,.2f}")

print("\n--- KEY INSIGHTS ---")
print("1. DEMOGRAPHICS: Younger users dominate Fashion/Electronics; Older users prefer Home/Grocery.")
print("2. REVENUE CONCENTRATION: Platinum & Gold members drive the highest transaction values.")
print("3. CHURN SIGNALS: Customers with long 'days_since_last_purchase' or low satisfaction scores are at high risk.")
print("4. RETENTION: Cohort analysis reveals a steep drop-off after the 3rd month.")
print("5. RETURNS: Fashion and Electronics experience the highest rate of returned orders.")

# Generate Summary Dashboard
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('E-Commerce KPI Dashboard', fontsize=24, color='white', weight='bold')

# Top Left: Revenue by Category
cat_rev = df.groupby('product_category')['total_amount'].sum().sort_values()
axes[0, 0].barh(cat_rev.index, cat_rev.values, color='cyan')
axes[0, 0].set_title('Revenue by Category')

# Top Right: Orders by Membership
mem_orders = df['membership_type'].value_counts()
axes[0, 1].pie(mem_orders, labels=mem_orders.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
axes[0, 1].set_title('Orders by Membership Tier')

# Bottom Left: Order Status
status_counts = df['order_status'].value_counts()
axes[1, 0].bar(status_counts.index, status_counts.values, color=['green', 'red', 'orange'])
axes[1, 0].set_title('Overall Order Status')

# Bottom Right: Discount vs Revenue
disc_rev = df.groupby('discount_pct')['total_amount'].sum()
axes[1, 1].plot(disc_rev.index, disc_rev.values, marker='o', color='yellow', linewidth=3)
axes[1, 1].set_title('Revenue vs Discount Percentage')
axes[1, 1].set_xlabel('Discount %')
axes[1, 1].set_ylabel('Total Revenue')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../outputs/17_summary_dashboard.png', bbox_inches='tight')
plt.close()

print("\n✅ Summary dashboard saved to outputs/17_summary_dashboard.png")
print("🎉 PROJECT EXECUTION COMPLETE!")
