# ============================================
# STAGE 2: Building & Training the ML Model
# ============================================
# We now take the raw data and teach a machine
# to predict which customers will leave.
#
# The algorithm we use: XGBoost
# Why XGBoost? It wins Kaggle competitions for
# tabular (table) data. Banks, telecom companies,
# fintechs all use it in production.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the trained model
import os

# ── LOAD DATA ──────────────────────────────────
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Loaded {len(df)} customers")

# ── DATA CLEANING ──────────────────────────────
# Remember Stage 1 showed TotalCharges is 'object' not float?
# That's because some rows have empty spaces instead of numbers.
# We must fix this — ML models only understand numbers.

# Step 1: Replace empty spaces with NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Step 2: How many NaN values did we get?
nan_count = df['TotalCharges'].isna().sum()
print(f"Rows with empty TotalCharges: {nan_count}")
# These are new customers (tenure=0) with no charges yet
# We fill them with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Step 3: Drop customerID — it's just an ID, not a pattern
# The model should learn from BEHAVIOUR, not arbitrary IDs
df = df.drop('customerID', axis=1)

# ── FEATURE ENGINEERING ────────────────────────
# "Features" = the inputs we give the model
# Right now many columns are text (Yes/No, Male/Female)
# ML models only understand numbers — we must convert

# Convert target variable: Churn Yes→1, No→0
# This is called "binary encoding" — 1 means churned, 0 means stayed
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"Churn encoded: {df['Churn'].value_counts().to_dict()}")

# Convert all other text columns to numbers
# LabelEncoder converts: Female→0, Male→1 etc.
# This is called "categorical encoding"
le = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"\nConverting {len(categorical_columns)} text columns to numbers:")
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    print(f"  {col}: converted")

# ── SPLIT DATA ─────────────────────────────────
# This is THE most important concept in ML:
# We split data into TRAINING set and TESTING set
#
# Training set: model LEARNS from this (like studying)
# Testing set:  we TEST the model on data it never saw (like an exam)
#
# If we tested on the same data we trained on,
# the model would just memorize answers — that's cheating!
# Real world performance = performance on UNSEEN data

X = df.drop('Churn', axis=1)  # X = all features (inputs)
y = df['Churn']                # y = what we're predicting (output)

# 80% training, 20% testing
# random_state=42 means we get the same split every time (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y means the 26.5% churn ratio is preserved in both splits
)

print(f"\nData split:")
print(f"  Training: {len(X_train)} customers (model learns from these)")
print(f"  Testing:  {len(X_test)} customers (model never sees these)")

# ── TRAIN THE MODEL ────────────────────────────
# XGBoost = Extreme Gradient Boosting
# It builds many small decision trees and combines them
# Each tree learns from the mistakes of the previous tree
# This is called "ensemble learning" — wisdom of crowds

print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,      # build 200 decision trees
    max_depth=4,           # each tree can be max 4 levels deep
    learning_rate=0.1,     # how much each tree corrects the previous
    subsample=0.8,         # use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,  # use 80% of features per tree
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)
print("Training complete!")

# ── EVALUATE THE MODEL ─────────────────────────
# Now we test on data the model NEVER saw
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of churning

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy:  {accuracy:.1%}")
print(f"AUC Score: {auc_score:.3f}")
print("\nWhat these mean:")
print(f"  Accuracy {accuracy:.1%} = model correctly identifies {accuracy:.1%} of customers")
print(f"  AUC {auc_score:.3f} = {auc_score:.1%} chance model ranks a churner higher than non-churner")
print("  AUC > 0.85 = production-ready model")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Stayed', 'Churned']))

# ── CONFUSION MATRIX ───────────────────────────
# This shows exactly where the model is right/wrong:
# True Positive:  predicted churn, actually churned ✓
# True Negative:  predicted stay, actually stayed ✓
# False Positive: predicted churn, actually stayed ✗ (false alarm)
# False Negative: predicted stay, actually churned ✗ (missed a churner)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Stay', 'Predicted Churn'],
            yticklabels=['Actually Stayed', 'Actually Churned'])
plt.title('Confusion Matrix — Where is the model right/wrong?',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/chart4_confusion_matrix.png')
plt.show()

# ── FEATURE IMPORTANCE ─────────────────────────
# Which features does the model rely on MOST?
# This is incredibly valuable for business insight
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature',
            palette='viridis')
plt.title('Top 10 Features — What drives churn?',
          fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('data/chart5_feature_importance.png')
plt.show()

print("\nTop 5 churn drivers:")
for _, row in importance_df.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# ── SAVE THE MODEL ─────────────────────────────
# Save the trained model so we can use it in the API later
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')
print("\nModel saved to models/churn_model.pkl")
print("\nSTAGE 2 COMPLETE — Ready for Stage 3: SHAP Explainability")