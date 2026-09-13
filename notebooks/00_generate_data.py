"""
E-Commerce Customer Behavior Analysis
Step 0: Generate Synthetic Dataset
Author: Amit Kumar

This script generates a realistic e-commerce transactions dataset with 1500+ rows.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

# 1. Setup Data Constants
NUM_CUSTOMERS = 400
NUM_TRANSACTIONS = 1600

CITIES = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad']
MEMBERSHIPS = ['Bronze', 'Silver', 'Gold', 'Platinum']
CATEGORIES = ['Electronics', 'Fashion', 'Home & Living', 'Beauty', 'Books', 'Sports', 'Grocery']
PAYMENT_METHODS = ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery', 'Wallet']
ORDER_STATUSES = ['Delivered', 'Returned', 'Cancelled']

# Generate realistic customer names (First + Last)
FIRST_NAMES = ['Rahul', 'Priya', 'Amit', 'Sneha', 'Vikram', 'Neha', 'Rohan', 'Pooja', 'Karan', 'Anjali', 
               'Aditya', 'Shruti', 'Sanjay', 'Kavita', 'Arjun', 'Riya', 'Mohit', 'Swati', 'Vishal', 'Aarti']
LAST_NAMES = ['Sharma', 'Singh', 'Patel', 'Kumar', 'Gupta', 'Verma', 'Reddy', 'Das', 'Joshi', 'Yadav',
              'Chauhan', 'Thakur', 'Bose', 'Nair', 'Menon', 'Mehta', 'Iyer', 'Bhatt', 'Rao', 'Desai']

# Products and prices per category
PRODUCTS = {
    'Electronics': [('Smartphone', 15000, 80000), ('Laptop', 35000, 120000), ('Headphones', 1000, 15000), ('Smartwatch', 2500, 25000)],
    'Fashion': [('T-Shirt', 300, 1500), ('Jeans', 800, 3500), ('Sneakers', 1500, 8000), ('Jacket', 1200, 5000)],
    'Home & Living': [('Bed Sheet', 400, 2000), ('Curtains', 600, 3000), ('Desk Lamp', 300, 1500), ('Cushions', 200, 1000)],
    'Beauty': [('Face Wash', 150, 800), ('Moisturizer', 250, 1200), ('Perfume', 500, 5000), ('Lipstick', 200, 1500)],
    'Books': [('Fiction Novel', 200, 800), ('Self Help', 150, 600), ('Biography', 250, 900), ('Comic', 100, 500)],
    'Sports': [('Yoga Mat', 300, 1500), ('Dumbbells', 500, 3000), ('Tennis Racket', 1000, 8000), ('Protein Powder', 1500, 4500)],
    'Grocery': [('Almonds', 400, 1200), ('Olive Oil', 300, 1000), ('Green Tea', 150, 500), ('Organic Honey', 250, 800)]
}

# 2. Generate Customer Base
customers = []
for i in range(1, NUM_CUSTOMERS + 1):
    c_id = f"C-{i:03d}"
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    age = random.randint(18, 65)
    gender = random.choice(['Male', 'Female'])
    city = random.choice(CITIES)
    
    # Older people tend to have higher memberships, but it's mixed
    mem_weights = [0.4, 0.3, 0.2, 0.1]
    if age > 40:
        mem_weights = [0.2, 0.3, 0.3, 0.2]
        
    membership = random.choices(MEMBERSHIPS, weights=mem_weights)[0]
    
    # Signup between Jan 2022 and Dec 2023
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_between = (end_date - start_date).days
    signup_date = start_date + timedelta(days=random.randint(0, days_between))
    
    customers.append({
        'customer_id': c_id,
        'customer_name': name,
        'age': age,
        'gender': gender,
        'city': city,
        'membership_type': membership,
        'signup_date': signup_date
    })

df_customers = pd.DataFrame(customers)

# 3. Generate Transactions
transactions = []
for i in range(1, NUM_TRANSACTIONS + 1):
    txn_id = f"TXN-{i:05d}"
    
    # Pick a random customer (weighted towards Platinum/Gold)
    cust = df_customers.sample(1, weights=df_customers['membership_type'].map({'Bronze':1, 'Silver':2, 'Gold':4, 'Platinum':6})).iloc[0]
    
    # Category Preference Rules
    cat_weights = [1]*len(CATEGORIES)
    if cust['age'] < 30:
        cat_weights[0] += 3  # Electronics
        cat_weights[1] += 3  # Fashion
    elif cust['age'] > 45:
        cat_weights[2] += 3  # Home & Living
        cat_weights[6] += 3  # Grocery
        
    if cust['gender'] == 'Female':
        cat_weights[3] += 2  # Beauty
        cat_weights[1] += 1  # Fashion
        
    category = random.choices(CATEGORIES, weights=cat_weights)[0]
    
    # Select product
    prod_tuple = random.choice(PRODUCTS[category])
    product_name = prod_tuple[0]
    unit_price = random.randint(prod_tuple[1], prod_tuple[2])
    
    # Sometimes inject an outlier price (1% chance)
    if random.random() < 0.01:
        unit_price = unit_price * random.randint(5, 10)
        
    quantity = random.randint(1, 5)
    discount_pct = random.choice([0, 5, 10, 15, 20, 25, 30])
    
    # Transaction date (after signup date, up to Dec 2024)
    max_txn_date = datetime(2024, 12, 31)
    if cust['signup_date'] < datetime(2023, 1, 1):
        txn_start = datetime(2023, 1, 1)
    else:
        txn_start = cust['signup_date']
        
    days_between = (max_txn_date - txn_start).days
    if days_between <= 0:
        txn_date = max_txn_date
    else:
        # Seasonality: more sales in Q4 (Oct, Nov, Dec)
        while True:
            t_date = txn_start + timedelta(days=random.randint(0, days_between))
            if t_date.month in [10, 11, 12]:
                if random.random() < 0.7: break
            else:
                if random.random() < 0.4: break
        txn_date = t_date

    payment = random.choice(PAYMENT_METHODS)
    
    # Order Status rules
    status_weights = [0.85, 0.10, 0.05] # Del, Ret, Can
    if category in ['Fashion', 'Electronics']:
        status_weights = [0.75, 0.20, 0.05]
        
    status = random.choices(ORDER_STATUSES, weights=status_weights)[0]
    
    # Satisfaction score
    if status == 'Returned':
        score = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
    elif status == 'Cancelled':
        score = random.choices([1, 2], weights=[0.7, 0.3])[0]
    else:
        score = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
        
    # Nullify some satisfaction scores
    if random.random() < 0.08:
        score = np.nan
        
    # Days since last purchase (churn signal)
    # Calculate difference from today (assume today is Jan 1, 2025)
    today = datetime(2025, 1, 1)
    days_since = (today - txn_date).days
    if random.random() < 0.05:
        days_since = np.nan
        
    transactions.append({
        'transaction_id': txn_id,
        'customer_id': cust['customer_id'],
        'customer_name': cust['customer_name'],
        'age': cust['age'],
        'gender': cust['gender'],
        'city': cust['city'],
        'membership_type': cust['membership_type'],
        'signup_date': cust['signup_date'].strftime('%Y-%m-%d'),
        'transaction_date': txn_date.strftime('%Y-%m-%d'),
        'product_category': category,
        'product_name': product_name,
        'quantity': quantity,
        'unit_price': unit_price,
        'discount_pct': discount_pct,
        'payment_method': payment,
        'order_status': status,
        'satisfaction_score': score,
        'days_since_last_purchase': days_since
    })

df = pd.DataFrame(transactions)

# 4. Inject Duplicate Transaction IDs for cleaning exercise
duplicates = df.sample(15)
df = pd.concat([df, duplicates], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

print(f"Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
df.to_csv('../data/ecommerce_transactions.csv', index=False)
print("Saved to data/ecommerce_transactions.csv")
