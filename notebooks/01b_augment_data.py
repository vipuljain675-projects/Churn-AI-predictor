import pandas as pd
import numpy as np

print("Injecting Synthetic Price Elasticity Data into IBM dataset...")

# Load original
try:
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Could not find original dataset.")
    exit(1)

# 1. Randomly assign a discount offered to every historical customer
np.random.seed(42)
discount_levels = [0, 5, 10, 15, 20]
probabilities = [0.60, 0.10, 0.10, 0.10, 0.10] 
df['Discount_Offered'] = np.random.choice(discount_levels, size=len(df), p=probabilities)

# 2. Simulate Price Elasticity (Human Psychology)
def adjust_churn(row):
    if row['Churn'] == 'No':
        return 'No'
    
    discount = row['Discount_Offered']
    if discount == 0:
        return 'Yes'
    elif discount == 5:
        # 10% chance a 5% discount works to retain them
        return 'No' if np.random.random() < 0.10 else 'Yes'
    elif discount == 10:
        # 30% chance a 10% discount works
        return 'No' if np.random.random() < 0.30 else 'Yes'
    elif discount == 15:
        # 60% chance a 15% discount works
        return 'No' if np.random.random() < 0.60 else 'Yes'
    elif discount == 20:
        # 85% chance a 20% discount works
        return 'No' if np.random.random() < 0.85 else 'Yes'
    return 'Yes'

df['Churn'] = df.apply(adjust_churn, axis=1)

# Save the augmented dataset
df.to_csv('data/WA_Fn-UseC_-Telco-Customer-Churn-Augmented.csv', index=False)
print("Success! Saved augmented dataset to 'data/WA_Fn-UseC_-Telco-Customer-Churn-Augmented.csv'")
