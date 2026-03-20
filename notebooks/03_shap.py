# ============================================
# STAGE 3: SHAP Explainability
# ============================================
# SHAP = SHapley Additive exPlanations
# 
# The problem with ML models: they're "black boxes"
# They say "this customer will churn" but not WHY
# 
# SHAP solves this — it explains EVERY prediction
# "This customer will churn BECAUSE:
#   - month-to-month contract (+0.4 churn risk)
#   - no online security (+0.2 churn risk)  
#   - only 3 months tenure (+0.15 churn risk)"
#
# This is what makes your project interview-worthy
# Any business person can understand SHAP output

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── LOAD MODEL AND DATA ────────────────────────
model = joblib.load('models/churn_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Recreate the same cleaned dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Calculating SHAP values... (30 seconds)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("Done!")

# ── CHART 1: GLOBAL FEATURE IMPORTANCE ─────────
# This shows OVERALL which features matter most
# across ALL customers — the big picture
print("\nGenerating Chart 1: Global feature importance...")
plt.figure()
shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("SHAP: Overall Feature Importance", fontweight='bold')
plt.tight_layout()
plt.savefig('data/shap_global.png', bbox_inches='tight')
plt.show()

# ── CHART 2: SHAP DOT PLOT ─────────────────────
# This is the most powerful chart — it shows:
# - Which features matter (y-axis)
# - How they affect churn (x-axis, red=increases churn)
# - The actual values (color: red=high, blue=low)
print("Generating Chart 2: SHAP dot plot...")
plt.figure()
shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Values: How each feature affects churn", fontweight='bold')
plt.tight_layout()
plt.savefig('data/shap_dot.png', bbox_inches='tight')
plt.show()

# ── EXPLAIN A SINGLE CUSTOMER ──────────────────
# This is the killer feature — explain ONE person
# Pick a high-risk customer (predicted to churn)
print("\n" + "="*50)
print("INDIVIDUAL CUSTOMER EXPLANATIONS")
print("="*50)

# Find a customer the model thinks WILL churn
y_pred_proba = model.predict_proba(X_test)[:, 1]
high_risk_idx = y_pred_proba.argmax()  # highest churn probability
high_risk_customer = X_test.iloc[high_risk_idx]
high_risk_prob = y_pred_proba[high_risk_idx]

print(f"\nHIGH RISK CUSTOMER (churn probability: {high_risk_prob:.1%})")
print("-" * 40)

# Get SHAP values for this customer
customer_shap = shap_values[high_risk_idx]

# Show top reasons WHY this customer will churn
reasons = pd.DataFrame({
    'feature': feature_names,
    'value': high_risk_customer.values,
    'shap_impact': customer_shap
}).sort_values('shap_impact', ascending=False)

print("TOP REASONS THIS CUSTOMER WILL CHURN:")
for _, row in reasons.head(5).iterrows():
    direction = "↑ INCREASES" if row['shap_impact'] > 0 else "↓ DECREASES"
    print(f"  {row['feature']} = {row['value']:.0f}  →  {direction} churn risk by {abs(row['shap_impact']):.3f}")

# ── BUSINESS INTERVENTION ──────────────────────
# The what-if simulator — what if we give a discount?
print("\n" + "="*50)
print("WHAT-IF SIMULATOR")
print("="*50)
print(f"Current churn probability: {high_risk_prob:.1%}")

# Simulate: what if we upgrade them to 1-year contract?
# Contract encoding: 0=Month-to-month, 1=One year, 2=Two year
modified_customer = high_risk_customer.copy()
modified_customer['Contract'] = 1  # upgrade to 1-year
new_prob = model.predict_proba(modified_customer.values.reshape(1, -1))[0][1]
print(f"If we upgrade to 1-year contract: {new_prob:.1%} churn probability")
print(f"Risk reduction: {(high_risk_prob - new_prob):.1%}")

# Calculate business value
monthly_charges = high_risk_customer['MonthlyCharges']
avg_tenure = 32  # months
saved_revenue = (high_risk_prob - new_prob) * monthly_charges * avg_tenure
print(f"Expected revenue saved: ${saved_revenue:.2f}")

print("\n" + "="*50)
print("STAGE 3 COMPLETE!")
print("="*50)
print("You now have:")
print("  ✓ Global feature importance (why churn happens)")
print("  ✓ Individual explanations (why THIS customer churns)")
print("  ✓ What-if simulator (what intervention saves them)")
print("\nNext: Stage 4 — FastAPI Backend")