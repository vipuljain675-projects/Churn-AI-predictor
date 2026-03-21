import os
import json
import joblib
import chromadb
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

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

print("Loading churn model...")
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df_raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0)
df_raw = df_raw.drop("customerID", axis=1)
df_raw["Churn"] = (df_raw["Churn"] == "Yes").astype(int)
for col in df_raw.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_raw[col] = le.fit_transform(df_raw[col])

print("Loading RAG database...")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("customers")
print(f"RAG database ready: {collection.count()} past customers")
print("Ready to chat!\n")

ROUTER_PROMPT = """You are a router for ChurnAI.
Decide: is the user describing a NEW customer, or asking a FOLLOW-UP about an already analyzed customer?

Reply with ONLY one word — NEW or FOLLOWUP.

NEW examples:
- "Customer, 2 months old, $90/month..."
- "I have a new customer..."

FOLLOWUP examples:
- "What if he leaves anyway?"
- "What is the revenue loss?"
- "What if we offer a discount?"
- "What else can we do?"
- "How long before he leaves?"
- "What if he goes to a competitor?"
- "Is annual contract the only option?"

Reply ONLY: NEW or FOLLOWUP"""

EXTRACTION_PROMPT = """You are ChurnAI. Extract customer features and output JSON inside <PREDICT> tags.
If info is missing, ask for it first.

<PREDICT>
{"tenure": X, "MonthlyCharges": X, "Contract": X, "OnlineSecurity": X,
 "TechSupport": X, "InternetService": X, "PaymentMethod": X,
 "SeniorCitizen": X, "Partner": X, "Dependents": X,
 "MultipleLines": X, "OnlineBackup": X, "DeviceProtection": X,
 "StreamingTV": X, "StreamingMovies": X, "PaperlessBilling": X,
 "TotalCharges": X, "PhoneService": X, "gender": X, "Discount_Offered": X}
</PREDICT>

Encodings:
- Contract: 0=Month-to-month, 1=One year, 2=Two year
- InternetService: 0=DSL, 1=Fiber optic, 2=No
- OnlineSecurity/TechSupport/OnlineBackup/DeviceProtection/StreamingTV/StreamingMovies: 0=No, 1=Yes, 2=No internet
- PaymentMethod: 0=Bank transfer, 1=Credit card, 2=Electronic check, 3=Mailed check
- gender: 0=Female, 1=Male
- All other yes/no: 0=No, 1=Yes
- TotalCharges = tenure x MonthlyCharges if not given"""

FOLLOWUP_PROMPT = """You are ChurnAI — a senior CFO and customer retention analyst.
Answer the follow-up question directly and specifically using the customer context.

CRITICAL ACCOUNTING RULES FOR PROFITABILITY:
If asked about profit margins, you MUST use this exact logic step-by-step:
1. Original Annual Revenue = (Monthly Revenue * 12)
2. Original Profit = Original Annual Revenue * (Profit Margin %)
3. Discount Amount = Original Annual Revenue * (Discount %)
4. New Discounted Revenue = Original Annual Revenue - Discount Amount
5. New Final Profit = New Discounted Revenue * (Profit Margin %)
Calculate step-by-step before answering to ensure perfect math."""

RECOMMENDATION_PROMPT = """You are ChurnAI. Give recommendation in EXACTLY this format:

WHAT'S HAPPENING:
[1 sentence — risk and main reason]

WHAT THE DATA SHOWS:
[1-2 sentences — past similar customers, what worked, what failed]

WHAT YOU MUST DO RIGHT NOW:
1. [Specific action with timeline. You must dynamically select the OPTIMAL discount percentage (e.g., 5%, 8%, or 15%) to offer. Do NOT default to 10%. Base your chosen % on their specific churn probability.]
2. [Specific offer using your chosen discount %]
3. [Backup if they refuse]

BUSINESS IMPACT:
[Explicitly calculate the financial impact using this strict formula:
1. Calculate Original Revenue (Monthly * 12)
2. Calculate Discount Amount
3. Calculate New Discounted Revenue
4. Calculate Final Profit = (New Discounted Revenue) * (20% Profit Margin).
Prove your chosen discount is profitable by stating the Final Profit.]

BOTTOM LINE:
[Single most important thing today]"""


def find_cohort_size(features, df):
    cohort = df.copy()
    cohort = cohort[cohort['Churn'] == 0] # Active customers only
    if 'Contract' in features:
        cohort = cohort[cohort['Contract'] == int(features['Contract'])]
    if 'InternetService' in features:
        cohort = cohort[cohort['InternetService'] == int(features['InternetService'])]
    if 'tenure' in features:
        # Group similar tenure within a 12 month band
        cohort = cohort[abs(cohort['tenure'] - features.get('tenure', 12)) <= 6]
    return max(1, len(cohort)) # Ensure at least 1 (this specific customer)

def customer_to_text(features):
    contract_map = {0: "month-to-month", 1: "one-year", 2: "two-year"}
    internet_map = {0: "DSL", 1: "fiber optic", 2: "no internet"}
    security_map = {0: "no security", 1: "has security", 2: "no internet"}
    support_map  = {0: "no tech support", 1: "has tech support", 2: "no internet"}
    return (
        f"Customer with {features.get('tenure', 12):.0f} months tenure, "
        f"paying {features.get('MonthlyCharges', 65):.0f} dollars monthly, "
        f"{contract_map.get(int(features.get('Contract', 0)), 'unknown')} contract, "
        f"{internet_map.get(int(features.get('InternetService', 1)), 'unknown')} internet, "
        f"{security_map.get(int(features.get('OnlineSecurity', 0)), 'unknown')}, "
        f"{support_map.get(int(features.get('TechSupport', 0)), 'unknown')}, "
        f"total charges {features.get('TotalCharges', 780):.0f} dollars"
    )


def find_similar_customers(features, n=5):
    results = collection.query(
        query_texts=[customer_to_text(features)],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    stayed  = [m for m in metadatas if m['outcome'] == 'STAYED']
    churned = [m for m in metadatas if m['outcome'] == 'CHURNED']
    worked  = list(set(m['intervention'] for m in stayed))
    failed  = list(set(m['intervention'] for m in churned))

    display = f"\n🔍 RAG EVIDENCE — {n} Most Similar Past Customers:\n" + "━"*50 + "\n"
    for i, (meta, dist) in enumerate(zip(metadatas, distances)):
        similarity = round((1 - dist) * 100, 1)
        icon = "✅" if meta['outcome'] == 'STAYED' else "❌"
        display += f"Customer {i+1} ({similarity}% similar): {icon} {meta['outcome']} | Tried: {meta['intervention'].replace('_',' ')} | Risk: {meta['predicted_risk']:.0%}\n"
    display += "━"*50 + "\n"
    display += f"📊 {len(stayed)}/5 STAYED | {len(churned)}/5 CHURNED\n"
    display += f"✅ Worked: {', '.join(worked) if worked else 'none'}\n"
    display += f"❌ Failed: {', '.join(failed) if failed else 'none'}\n"

    summary = (
        f"{len(stayed)}/5 similar customers STAYED, {len(churned)}/5 CHURNED. "
        f"Worked: {', '.join(worked) if worked else 'none'}. "
        f"Failed: {', '.join(failed) if failed else 'none'}."
    )
    return display, summary


def run_prediction(features):
    defaults = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 2, "MonthlyCharges": 65.0, "TotalCharges": 780.0,
        "Discount_Offered": 0
    }
    defaults.update(features)
    row = pd.DataFrame([[defaults.get(f, 0) for f in feature_names]], columns=feature_names)
    prob = model.predict_proba(row)[0][1]
    risk = "🔴 HIGH RISK" if prob > 0.75 else "🟡 MEDIUM RISK" if prob > 0.5 else "🟢 LOW RISK"
    row_mod = row.copy()
    row_mod["Contract"] = 1
    prob_c = model.predict_proba(row_mod)[0][1]
    savings = (prob - prob_c) * defaults["MonthlyCharges"] * 32
    return {"probability": prob, "risk": risk, "contract_prob": prob_c,
            "savings": savings, "monthly": defaults["MonthlyCharges"], "tenure": defaults["tenure"]}


def add_to_rag(features, outcome, intervention):
    import time
    collection.add(
        documents=[customer_to_text(features)],
        metadatas=[{"customer_index": f"new_{int(time.time())}", "actually_churned": 1 if outcome=="CHURNED" else 0,
                    "predicted_risk": 0.0, "outcome": outcome, "intervention": intervention,
                    "monthly_charges": float(features.get("MonthlyCharges", 65)),
                    "tenure": float(features.get("tenure", 12)), "contract": float(features.get("Contract", 0))}],
        ids=[f"new_{int(time.time())}"]
    )
    print(f"\n✅ Added to RAG. Total: {collection.count()} customers")


def chat():
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("="*55)
    print("   🤖 CHURN AI WITH RAG — Evidence-Based Retention")
    print("="*55)
    print("Describe a customer → get analysis")
    print("Then ask follow-up questions freely!")
    print("Record outcome: 'outcome: STAYED, intervention: annual_contract'")
    print("Type 'exit' to quit.")
    print("="*55 + "\n")

    chat_history    = []
    last_features   = None
    current_context = None  # ← stores full customer context for follow-ups

    while True:
        user_input = input("\nYou → ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("ChurnAI → Goodbye! 👋")
            break

        if user_input.lower().startswith("outcome:") and last_features:
            parts = user_input.split(",")
            outcome = parts[0].split(":")[1].strip().upper()
            intervention = parts[1].split(":")[1].strip() if len(parts) > 1 else "unknown"
            add_to_rag(last_features, outcome, intervention)
            continue

        # ── ROUTE: new customer or follow-up? ──
        route = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": ROUTER_PROMPT},
                      {"role": "user", "content": user_input}],
            max_tokens=5, temperature=0.0,
        ).choices[0].message.content.strip().upper()

        # ── FOLLOW-UP: answer about current customer ──
        if "FOLLOWUP" in route and current_context:
            answer = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": FOLLOWUP_PROMPT},
                    {"role": "user", "content": f"CUSTOMER CONTEXT:\n{current_context}\n\nQUESTION: {user_input}"}
                ],
                max_tokens=1200, temperature=0.7,
            ).choices[0].message.content.strip()
            print(f"\nChurnAI → {answer}")
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})
            continue

        # ── NEW CUSTOMER: extract + predict + recommend ──
        chat_history.append({"role": "user", "content": user_input})

        ai_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": EXTRACTION_PROMPT}] + chat_history[-8:],
            max_tokens=512, temperature=0.3,
        ).choices[0].message.content.strip()

        if "<PREDICT>" in ai_response and "</PREDICT>" in ai_response:
            json_str = ai_response[ai_response.index("<PREDICT>")+9:ai_response.index("</PREDICT>")].strip()
            try:
                features      = json.loads(json_str)
                last_features = features
                result        = run_prediction(features)
                rag_display, rag_summary = find_similar_customers(features)
                cohort_size   = find_cohort_size(features, df_raw)

                # Pure Business Math (Revenue Only - Let LLM Dynamically Create Discounts and Profit Margins)
                revenue_per_user     = result['monthly'] * 12 # 12-month forward ARR
                
                # Global scale
                global_revenue_loss  = cohort_size * revenue_per_user

                # Generate true ML elasticity curve
                elasticity_text = "PRICE ELASTICITY CURVE (Annual Contract Churn Risk by Discount %):\n"
                for d in [0, 5, 10, 15, 20]:
                    r = features.copy()
                    r["Contract"] = 1 # One-year
                    r["Discount_Offered"] = d
                    res_d = run_prediction(r)
                    elasticity_text += f"- {d}% discount: {res_d['probability']:.1%} risk\n"

                recommendation = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": RECOMMENDATION_PROMPT},
                        {"role": "user", "content": (
                            f"Churn probability: {result['probability']:.1%} ({result['risk']})\n"
                            f"Monthly Revenue: ${result['monthly']:.0f}\n"
                            f"{elasticity_text}\n"
                            f"COHORT SIZE: {cohort_size} active customers in the database share this profile.\n"
                            f"GLOBAL REVENUE RISK: If all {cohort_size} churn, we lose ${global_revenue_loss:,.0f} in ARR.\n"
                            f"RAG: {rag_summary}"
                        )}
                    ],
                    max_tokens=1200, temperature=0.5,
                ).choices[0].message.content.strip()

                # Store full context so follow-ups work
                current_context = (
                    f"Profile: {customer_to_text(features)}\n"
                    f"Churn risk: {result['probability']:.1%} ({result['risk']})\n"
                    f"Monthly charges: ${result['monthly']:.0f}\n"
                    f"{elasticity_text}\n"
                    f"COHORT SIZE: {cohort_size} active customers match this profile.\n"
                    f"GLOBAL REVENUE RISK (All {cohort_size} churn): We lose ${global_revenue_loss:,.0f} in Top-Line Revenue.\n"
                    f"RAG Evidence: {rag_summary}\n"
                    f"Recommendation: {recommendation}"
                )

                print(f"\nChurnAI → \n{'━'*50}\n📊 XGBOOST PREDICTION\n{'━'*50}")
                print(f"Result:      {result['risk']}")
                print(f"Probability: {result['probability']:.1%} chance of churning")
                print(f"Contract upgrade → {result['contract_prob']:.1%} | Saves ${result['savings']:.2f}")
                print(rag_display)
                print(f"{'━'*50}\n🤖 CHURNAI RECOMMENDATION\n{'━'*50}")
                print(recommendation)
                print(f"{'━'*50}")
                print("💬 Ask me anything about this customer!")
                print("   Or: outcome: STAYED, intervention: annual_contract")
                print(f"{'━'*50}")

                chat_history.append({"role": "assistant", "content": f"Analysis complete. Risk: {result['risk']}. {recommendation}"})

            except Exception as e:
                print(f"\nChurnAI → Error: {e}")
        else:
            print(f"\nChurnAI → {ai_response}")
            chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    chat()