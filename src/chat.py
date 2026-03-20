# ============================================
# STAGE 4: AI Chat Interface for Churn Model
# ============================================
# This combines:
# 1. Your trained XGBoost model (the brain)
# 2. Groq LLaMA 70B (the voice)
# 3. A chat interface (the face)
#
# User can now TALK to the model in plain English
# instead of filling spreadsheets

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── LOAD ENV ───────────────────────────────────
def load_env():
    env_path = Path.home() / "Documents" / "ML PROJECT" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()

# ── LOAD MODEL ─────────────────────────────────
print("Loading churn model...")
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Recreate label encoders to decode values later
df_raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw["TotalCharges"] = pd.to_numeric(
    df_raw["TotalCharges"], errors="coerce"
).fillna(0)
df_raw = df_raw.drop("customerID", axis=1)
df_raw["Churn"] = (df_raw["Churn"] == "Yes").astype(int)

encoders = {}
for col in df_raw.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_raw[col] = le.fit_transform(df_raw[col])
    encoders[col] = le

print("Model loaded. Ready to chat!")

# ── SYSTEM PROMPT ──────────────────────────────
SYSTEM_PROMPT = """You are ChurnAI — an expert customer retention analyst.
You have access to a trained XGBoost churn prediction model.

Your job:
1. Help users understand why customers churn
2. Predict churn risk for specific customers when given their details
3. Suggest specific retention interventions
4. Answer questions about churn patterns in plain English

When a user gives you customer details, extract these features:
- tenure (months as customer, number)
- MonthlyCharges (monthly bill in USD, number)  
- Contract (0=Month-to-month, 1=One year, 2=Two year)
- OnlineSecurity (0=No, 1=Yes, 2=No internet service)
- TechSupport (0=No, 1=Yes, 2=No internet service)
- InternetService (0=DSL, 1=Fiber optic, 2=No)
- PaymentMethod (0=Bank transfer, 1=Credit card, 2=Electronic check, 3=Mailed check)
- SeniorCitizen (0=No, 1=Yes)
- Partner (0=No, 1=Yes)
- Dependents (0=No, 1=Yes)

When you have enough info to make a prediction, respond with a JSON block like:
<PREDICT>
{"tenure": 5, "MonthlyCharges": 80, "Contract": 0, "OnlineSecurity": 0, 
 "TechSupport": 0, "InternetService": 1, "PaymentMethod": 2, 
 "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
 "MultipleLines": 1, "OnlineBackup": 0, "DeviceProtection": 0,
 "StreamingTV": 0, "StreamingMovies": 0, "PaperlessBilling": 1,
 "TotalCharges": 400, "PhoneService": 1, "gender": 1}
</PREDICT>

Always explain results in business terms with specific retention actions.
Be conversational, direct, and actionable. No fluff."""


def run_model_prediction(features: dict) -> dict:
    """Run the actual XGBoost model on extracted features."""
    # Build a full feature vector with defaults for missing features
    defaults = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 2, "MonthlyCharges": 65.0, "TotalCharges": 780.0
    }
    defaults.update(features)

    # Build DataFrame in correct feature order
    row = pd.DataFrame([[defaults.get(f, 0) for f in feature_names]],
                       columns=feature_names)

    prob = model.predict_proba(row)[0][1]
    prediction = "WILL CHURN" if prob > 0.5 else "WILL STAY"

    # Risk level
    if prob > 0.75:
        risk = "🔴 HIGH RISK"
    elif prob > 0.5:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    # What-if: upgrade to annual contract
    row_modified = row.copy()
    row_modified["Contract"] = 1
    prob_with_contract = model.predict_proba(row_modified)[0][1]
    savings = (prob - prob_with_contract) * defaults["MonthlyCharges"] * 32

    return {
        "probability": prob,
        "prediction": prediction,
        "risk_level": risk,
        "contract_intervention_prob": prob_with_contract,
        "expected_savings": savings
    }


def chat():
    """Main chat loop."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("\n" + "="*55)
    print("   🤖 CHURN AI — Customer Retention Assistant")
    print("="*55)
    print("Ask me anything about customer churn!")
    print("Examples:")
    print("  'Why do customers churn?'")
    print("  'I have a customer, 2 months old, paying $85/month")
    print("   on month-to-month, no security. Will they leave?'")
    print("  'What's the best way to retain fiber optic customers?'")
    print("\nType 'exit' to quit.")
    print("="*55 + "\n")

    chat_history = []

    while True:
        user_input = input("\nYou → ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("ChurnAI → Goodbye! Keep those customers happy. 👋")
            break

        # Add to history
        chat_history.append({"role": "user", "content": user_input})

        # Call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                     + chat_history[-6:],  # last 3 turns
            max_tokens=1024,
            temperature=0.7,
        )

        ai_response = response.choices[0].message.content.strip()

        # Check if model wants to make a prediction
        if "<PREDICT>" in ai_response and "</PREDICT>" in ai_response:
            # Extract JSON between tags
            start = ai_response.index("<PREDICT>") + len("<PREDICT>")
            end = ai_response.index("</PREDICT>")
            json_str = ai_response[start:end].strip()

            try:
                features = json.loads(json_str)
                result = run_model_prediction(features)

                # Replace the PREDICT block with actual results
                prediction_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 MODEL PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result:      {result['risk_level']} — {result['prediction']}
Probability: {result['probability']:.1%} chance of churning

💡 INTERVENTION SIMULATION:
If upgraded to 1-year contract:
  → Churn probability drops to {result['contract_intervention_prob']:.1%}
  → Expected revenue saved: ${result['expected_savings']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

                # Remove the JSON block from response
                clean_response = (ai_response[:ai_response.index("<PREDICT>")]
                                  + prediction_block
                                  + ai_response[ai_response.index("</PREDICT>")
                                                + len("</PREDICT>"):])
                ai_response = clean_response

            except Exception as e:
                pass  # If parsing fails, just show the raw response

        print(f"\nChurnAI → {ai_response}")
        chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    chat()