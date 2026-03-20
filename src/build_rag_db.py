# ============================================
# BUILD RAG DATABASE — Run this ONCE only
# ============================================
# Takes our 1,409 test customers (from Stage 2)
# Converts each to a vector
# Stores in ChromaDB with their actual outcome
# This becomes our "memory" of past customers

import pandas as pd
import numpy as np
import joblib
import chromadb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── LOAD EVERYTHING WE ALREADY HAVE ───────────
print("Loading existing model and data...")
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Recreate same dataset as Stage 2
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

# Same split as Stage 2 — IDENTICAL random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get model's predicted probabilities for test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Test customers to store: {len(X_test)}")

# ── SETUP CHROMADB ─────────────────────────────
# ChromaDB is a local vector database
# It stores vectors + metadata on your laptop
# No API key, no internet, completely free
print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path="chroma_db")

# Delete existing collection if rebuilding
try:
    client.delete_collection("customers")
    print("Cleared existing collection.")
except:
    pass

# Create fresh collection
collection = client.create_collection(
    name="customers",
    # ChromaDB has built-in embedding — we use it
    # It converts text to vectors automatically
    metadata={"hnsw:space": "cosine"}
)

# ── CONVERT CUSTOMERS TO TEXT + STORE ─────────
# We convert each customer profile to a sentence
# ChromaDB converts that sentence to a vector
# This is the embedding step

def customer_to_text(row, feature_names):
    """Convert customer features to descriptive text."""
    # Map encoded numbers back to readable labels
    contract_map = {0: "month-to-month", 1: "one-year", 2: "two-year"}
    internet_map = {0: "DSL", 1: "fiber optic", 2: "no internet"}
    security_map = {0: "no security", 1: "has security", 2: "no internet"}
    support_map  = {0: "no tech support", 1: "has tech support", 2: "no internet"}

    features = dict(zip(feature_names, row))

    text = (
        f"Customer with {features['tenure']:.0f} months tenure, "
        f"paying {features['MonthlyCharges']:.0f} dollars monthly, "
        f"{contract_map.get(int(features['Contract']), 'unknown')} contract, "
        f"{internet_map.get(int(features['InternetService']), 'unknown')} internet, "
        f"{security_map.get(int(features['OnlineSecurity']), 'unknown')}, "
        f"{support_map.get(int(features['TechSupport']), 'unknown')}, "
        f"total charges {features['TotalCharges']:.0f} dollars, "
        f"senior citizen: {bool(features['SeniorCitizen'])}, "
        f"has partner: {bool(features['Partner'])}, "
        f"has dependents: {bool(features['Dependents'])}"
    )
    return text

print("Converting customers to text and storing in ChromaDB...")
print("This takes ~30 seconds...")

# Store in batches of 100 for speed
batch_size = 100
documents = []
metadatas = []
ids = []

for i, (idx, row) in enumerate(X_test.iterrows()):
    # Convert to text
    text = customer_to_text(row.values, feature_names)
    
    # Metadata = everything we know about outcome
    actually_churned = int(y_test.iloc[i])
    predicted_risk = float(y_pred_proba[i])
    
    # Determine what "intervention" worked
    # In real world this would be actual retention action taken
    # Here we simulate based on outcome + risk
    if actually_churned == 0 and predicted_risk > 0.5:
        # Model thought they'd churn but they stayed
        # Simulate: they were offered something that worked
        intervention = "annual_contract_offer"
        outcome = "STAYED"
    elif actually_churned == 0 and predicted_risk <= 0.5:
        # Low risk, stayed — no intervention needed
        intervention = "no_intervention_needed"
        outcome = "STAYED"
    elif actually_churned == 1 and predicted_risk > 0.5:
        # High risk, churned — intervention failed or not tried
        intervention = "discount_offer_failed"
        outcome = "CHURNED"
    else:
        # Low risk but churned — surprise churn
        intervention = "no_intervention_churned"
        outcome = "CHURNED"

    documents.append(text)
    metadatas.append({
        "customer_index": str(i),
        "actually_churned": actually_churned,
        "predicted_risk": round(predicted_risk, 3),
        "outcome": outcome,
        "intervention": intervention,
        "monthly_charges": float(row['MonthlyCharges']),
        "tenure": float(row['tenure']),
        "contract": float(row['Contract']),
    })
    ids.append(f"customer_{i}")

    # Store in batches
    if len(documents) == batch_size:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        documents = []
        metadatas = []
        ids = []
        print(f"  Stored {i+1}/{len(X_test)} customers...")

# Store remaining
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"\n✅ RAG Database built!")
print(f"   Total customers stored: {collection.count()}")
print(f"   Location: chroma_db/")
print(f"\nBreakdown:")
stayed  = sum(1 for m in collection.get()['metadatas'] if m['outcome'] == 'STAYED')
churned = sum(1 for m in collection.get()['metadatas'] if m['outcome'] == 'CHURNED')
print(f"   Stayed:  {stayed}")
print(f"   Churned: {churned}")
print(f"\nNext: run python3 src/chat_with_rag.py")