# ============================================
# STAGE 1: Understanding Our Data
# ============================================
# Before ANY machine learning, we must understand
# what we're working with. This is called EDA —
# Exploratory Data Analysis. Every data scientist
# does this first, always.

import pandas as pd        # pandas = tool for working with tables of data
import numpy as np         # numpy = tool for math operations
import matplotlib.pyplot as plt  # for drawing charts
import seaborn as sns      # makes charts look better

# ── LOAD THE DATA ──────────────────────────────
# pd.read_csv reads a CSV file into a "DataFrame"
# A DataFrame is just a table — rows = customers, columns = features
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# ── FIRST LOOK ─────────────────────────────────
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
# .shape tells us (number of rows, number of columns)
# rows = customers, columns = features about each customer
print(f"Rows (customers): {df.shape[0]}")
print(f"Columns (features): {df.shape[1]}")

print("\n" + "=" * 50)
print("FIRST 5 CUSTOMERS")
print("=" * 50)
# .head() shows first 5 rows — always do this first
# to understand what your data actually looks like
print(df.head())

print("\n" + "=" * 50)
print("COLUMN NAMES & DATA TYPES")
print("=" * 50)
# .dtypes tells you what TYPE each column is
# object = text, int64 = whole number, float64 = decimal number
print(df.dtypes)

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
# .isnull().sum() counts empty/missing cells per column
# Missing data = poison for ML models, must handle it
missing = df.isnull().sum()
print(missing[missing > 0])  # only show columns that HAVE missing values
if missing.sum() == 0:
    print("No missing values! Clean dataset.")

print("\n" + "=" * 50)
print("CHURN DISTRIBUTION — THE KEY QUESTION")
print("=" * 50)
# This is the most important cell — how many customers
# actually churn vs stay? This is what we're predicting.
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"Stayed:  {churn_counts['No']}  ({churn_pct['No']:.1f}%)")
print(f"Churned: {churn_counts['Yes']} ({churn_pct['Yes']:.1f}%)")
print("\nIMPORTANT: If churn % is very low (< 5%), we have")
print("a 'class imbalance' problem we need to handle.")

print("\n" + "=" * 50)
print("NUMERICAL FEATURES SUMMARY")
print("=" * 50)
# .describe() gives stats for number columns:
# mean, min, max, std (standard deviation = how spread out values are)
print(df.describe())

# ── VISUALIZATIONS ─────────────────────────────
print("\nGenerating charts... (check the popup windows)")

# Chart 1: Churn distribution — simple bar chart
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('How many customers churned?', fontsize=14, fontweight='bold')
plt.xlabel('Churn (No = Stayed, Yes = Left)')
plt.ylabel('Number of Customers')
# Add numbers on bars so it's readable
for i, count in enumerate(churn_counts):
    plt.text(i, count + 50, str(count), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('data/chart1_churn_distribution.png')
plt.show()
print("Chart 1 saved.")

# Chart 2: Monthly charges vs Churn
# Do customers who pay more tend to leave more? Let's see.
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn',
             bins=30, palette=['#2ecc71', '#e74c3c'], alpha=0.7)
plt.title('Monthly Charges: Do higher-paying customers leave more?',
          fontsize=13, fontweight='bold')
plt.xlabel('Monthly Charges (USD)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('data/chart2_monthly_charges.png')
plt.show()
print("Chart 2 saved.")

# Chart 3: Contract type vs Churn
# Month-to-month vs 1 year vs 2 year contracts — which churns most?
plt.figure(figsize=(10, 5))
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
contract_churn.plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                    figsize=(10, 5), edgecolor='black')
plt.title('Contract Type vs Churn — The Biggest Pattern',
          fontsize=13, fontweight='bold')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(['Stayed', 'Churned'])
plt.tight_layout()
plt.savefig('data/chart3_contract_churn.png')
plt.show()
print("Chart 3 saved.")

print("\n" + "=" * 50)
print("STAGE 1 COMPLETE")
print("=" * 50)
print("You now know:")
print("1. How many customers are in the dataset")
print("2. How many features describe each customer")
print("3. What % of customers churn")
print("4. Which features likely matter most")
print("\nNext: Stage 2 — Train the ML model")