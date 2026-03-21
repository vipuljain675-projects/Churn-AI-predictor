# ChurnAI — Customer Retention Intelligence System

> Built by Vipul Jain | 2nd Semester CSE | Built in 1 day

A production-grade AI system that predicts customer churn, explains WHY using SHAP, retrieves evidence from 1,409 past customers using RAG, and gives evidence-based retention recommendations through a conversational chat interface.

---

## The Business Problem This Solves

Every telecom company loses customers every month. The problem is not losing them — it's **not knowing WHO will leave BEFORE they leave.**

```
Without ChurnAI:
  Customer leaves → company notices → too late
  Revenue lost: $95/month × 12 months = $1,140 per customer
  1,000 customers leave/month = $1,140,000 lost monthly

With ChurnAI:
  System flags customer BEFORE they leave
  Retention team calls them within 24 hours
  Offers annual contract
  Customer stays
  Revenue saved: $1,402 per high-risk customer
```

---

## Built For — Telco Company Use Case

Built specifically for a **telecom company** using the IBM Telco Customer Churn dataset.

### Company-Specific Features

| Feature | What It Means | Why It Matters |
|---------|--------------|----------------|
| `tenure` | Months with company | New customers churn most — first 3 months critical |
| `MonthlyCharges` | Monthly bill | High bill + no value = leaves |
| `Contract` | Month-to-month/1yr/2yr | #1 churn driver — month-to-month = 42% churn rate |
| `InternetService` | DSL/Fiber/None | Fiber customers churn MORE — more competition |
| `OnlineSecurity` | Security bundle | No security = feels unsafe = leaves |
| `TechSupport` | Support plan | No support = frustrated = leaves |
| `PaymentMethod` | How they pay | Electronic check customers churn most |
| `TotalCharges` | Total paid since joining | Low total = new customer = high risk |
| `Partner/Dependents` | Family status | Family = more stable = stays |

### What the Model Discovered

```
Top 5 churn drivers (SHAP analysis):
1. Contract type     38.4% importance ← BIGGEST factor
2. Tenure            how long with company
3. MonthlyCharges    bill amount
4. OnlineSecurity    has security or not
5. TechSupport       has support or not

Business insight discovered by pure data:
Month-to-month + new customer + high bill + no services
= ticking time bomb.
Lock them into annual contract within first 3 months
or you WILL lose them.
```

### Domain-Agnostic — Works for Any Company

Only the CSV changes. Everything else is identical:

```
Telecom  → tenure, contract, monthly charges    (this project)
Swiggy   → orders/month, last order date, avg order value
Netflix  → watch hours, days since login, shows completed
Bank     → account age, transactions/month, balance
SaaS     → logins/month, features used, support tickets
```

---

## Architecture

```
USER types customer description
        ↓
ROUTER (LLaMA 70B, temp=0.0)
"NEW customer or FOLLOW-UP question?"
        ↓                    ↓
      NEW               FOLLOW-UP
        ↓                    ↓
EXTRACTION            Answer using
(LLaMA 70B,           current_context
temp=0.3)             variable
        ↓
  JSON features
        ↓
TWO PARALLEL:
XGBoost (laptop)    ChromaDB RAG
predicts churn %    finds 5 similar
0.01 seconds        past customers
no internet         no internet
        ↓                ↓
RECOMMENDATION ENGINE (LLaMA 70B, temp=0.5)
combines prediction + RAG evidence
outputs structured advice with business math
        ↓
CONTEXT STORED → enables follow-up questions
```

---

## Project Structure — Every File Explained

```
churn-predictor/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← 7,043 customers (SOURCE)
│   ├── chart1_churn_distribution.png          ← from 01_explore.py
│   ├── chart4_confusion_matrix.png            ← from 02_model.py
│   ├── chart5_feature_importance.png          ← from 02_model.py
│   ├── shap_global.png                        ← from 03_shap.py
│   └── shap_dot.png                           ← from 03_shap.py
│
├── models/
│   ├── churn_model.pkl      ← trained XGBoost (200 trees, the brain)
│   └── feature_names.pkl    ← ['tenure','MonthlyCharges'...] 19 names
│
├── notebooks/               ← run ONCE on laptop, never on Colab
│   ├── 01_explore.py        ← Stage 1: understand the data
│   ├── 02_model.py          ← Stage 2: train + save XGBoost
│   └── 03_shap.py           ← Stage 3: explain predictions
│
├── src/                     ← the actual product
│   ├── build_rag_db.py      ← builds ChromaDB (run ONCE)
│   └── chat_with_rag.py     ← THE MAIN APP (run daily)
│
└── chroma_db/               ← RAG DATABASE (auto-created)
    ├── data_level0.bin      ← 1,409 customer vectors (binary)
    ├── index_metadata.pickle← HNSW index config
    ├── link_lists.bin       ← HNSW graph for fast search
    └── chroma.sqlite3       ← outcomes + interventions (readable)
```

---

## Code Walkthrough — Every File, Every Function

### `notebooks/01_explore.py` — Understand the Data

Run once. Never goes to Colab. Just pandas + charts on your laptop.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV into DataFrame (table in Python)
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# df.shape → (7043, 21) — 7,043 rows, 21 columns

# Check missing values
df.isnull().sum()
# isnull() marks empty cells True, .sum() counts them
# Result: 0 missing (but TotalCharges has hidden spaces)

# Check churn distribution
df['Churn'].value_counts()
# No=5174 (73.5%), Yes=1869 (26.5%)
# 26.5% churn = healthy balance for ML training

# Chart: who churns by contract type
df.groupby(['Contract', 'Churn']).size().unstack().plot(kind='bar')
# Reveals: month-to-month = 42% churn, 2-year = 3% churn
# This single finding is worth millions to the company
```

**Key insight:** Month-to-month customers churn 14x more than 2-year customers.

---

### `notebooks/02_model.py` — Train XGBoost

Run once. Produces `models/churn_model.pkl`.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import joblib

# Fix broken column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# TotalCharges stored as TEXT with spaces in some rows
# errors='coerce' → convert to NaN instead of crashing
df['TotalCharges'].fillna(0)
# New customers have no charges yet → fill with 0

# Encode text columns to numbers
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
# "Month-to-month"→0, "One year"→1, "Two year"→2
# XGBoost only understands numbers, never text
# fit() learns the mapping, transform() applies it

# Split 80/20
X = df.drop('Churn', axis=1)  # 19 features = inputs
y = df['Churn']               # Churn column = output

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% = 1,409 customers for testing
    random_state=42,  # same split every run (reproducible)
    stratify=y        # preserve 26.5% churn in both splits
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,     # 200 decision trees
    max_depth=4,          # each tree asks max 4 questions
    learning_rate=0.1,    # each tree corrects 10% of previous error
    subsample=0.8,        # each tree sees 80% of data (no memorizing)
    colsample_bytree=0.8  # each tree sees 80% of features (no memorizing)
)
model.fit(X_train, y_train)
# Takes 3 seconds on MacBook — no GPU needed

# Evaluate
y_pred = model.predict(X_test)          # 0 or 1
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 0.0 to 1.0
accuracy = accuracy_score(y_test, y_pred)  # 0.80
auc = roc_auc_score(y_test, y_pred_proba)  # 0.841

# Save to disk — so we never retrain again
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')
```

---

### `notebooks/03_shap.py` — Explain Predictions

Run once. Answers WHY model predicts what it predicts.

```python
import shap

# Create SHAP explainer for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# shap_values[customer][feature] = impact on churn probability
# Positive = increased churn risk
# Negative = decreased churn risk

# Global chart — what matters overall
shap.summary_plot(shap_values, X_test, plot_type="bar")
# Contract: 0.84 average impact (biggest)
# Tenure: 0.52 average impact (second)

# Dot plot — how each feature affects each customer
shap.summary_plot(shap_values, X_test)
# Blue Contract dots far RIGHT = month-to-month customers at HIGH risk
# Red Tenure dots far LEFT = long-tenure customers at LOW risk

# Single customer explanation
customer_shap = shap_values[high_risk_idx]
# tenure=1 → +1.017 (1 month tenure = massive risk)
# Contract=0 → +0.777 (month-to-month = high risk)
# No security → +0.294 (no security bundle = higher risk)

# What-if simulation
row_modified['Contract'] = 1  # upgrade to annual
new_prob = model.predict_proba(row_modified)[0][1]
# 95.4% churn → 63.7% churn just by changing contract
savings = (0.954 - 0.637) * 90 * 32  # = $912 saved
```

---

### `src/build_rag_db.py` — Build RAG Database

Run once. Creates `chroma_db/` with 1,409 customer vectors.

```python
import chromadb

# Setup ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
# Saves to disk — survives computer restarts
# Creates chroma_db/ folder automatically

collection = client.create_collection(
    name="customers",
    metadata={"hnsw:space": "cosine"}
    # cosine similarity = compare vector DIRECTIONS not distances
)

# Same split as 02_model.py — CRITICAL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# random_state=42 guarantees IDENTICAL 1,409 customers
# every single time you run this

# Convert customer to text for embedding
def customer_to_text(row, feature_names):
    return (
        f"Customer with {tenure} months tenure, "
        f"paying {charges} dollars monthly, "
        f"{contract} contract, {internet} internet..."
    )
# Why text not numbers?
# Embedding models understand semantic meaning of text
# "month-to-month" carries more meaning than "0"

# Store each customer
collection.add(
    documents=[text],        # ChromaDB embeds this to 384 numbers
    metadatas=[{
        "outcome": "CHURNED",
        "intervention": "discount_offer_failed",
        "predicted_risk": 0.78,
        "monthly_charges": 90.0
    }],
    ids=["customer_0"]
)
# ChromaDB automatically:
# 1. Runs text through embedding model → 384-dim vector
# 2. Stores vector in data_level0.bin
# 3. Stores metadata in chroma.sqlite3
# 4. Updates HNSW search index
```

**How 20% became RAG data:**

```
7,043 customers in CSV
        ↓ train_test_split(test_size=0.2, random_state=42)
5,634 (80%) → XGBoost learned patterns from these
1,409 (20%) → became RAG database

Why test set specifically?
We know EXACTLY what happened to these 1,409 customers:
  - What XGBoost predicted (risk score)
  - What actually happened (churned or stayed)
  - What could have been tried (intervention)
This is REAL past evidence — perfect for RAG
```

---

### `src/chat_with_rag.py` — The Main Application

Run daily. The complete product.

```python
# FOUR PROMPTS — each has one job:

ROUTER_PROMPT = "Reply ONLY: NEW or FOLLOWUP"
# temperature=0.0 → always deterministic
# Binary decision — zero randomness allowed

EXTRACTION_PROMPT = "Extract features, output JSON in <PREDICT> tags"
# temperature=0.3 → mostly consistent
# Slight flexibility because customers describe info differently

RECOMMENDATION_PROMPT = "Give advice in exact format: WHAT'S HAPPENING / DATA SHOWS / DO NOW / BOTTOM LINE"
# temperature=0.5 → balanced
# Consistent format but tailored content

FOLLOWUP_PROMPT = "Answer follow-up using stored customer context"
# temperature=0.7 → conversational
# Natural dialogue, not templated

# MAIN LOOP:
while True:
    user_input = input("You → ")

    # Route: new or follow-up?
    route = groq.chat(ROUTER_PROMPT, user_input, temp=0.0)

    if "FOLLOWUP" in route and current_context:
        # Answer about SAME customer using stored context
        answer = groq.chat(FOLLOWUP_PROMPT,
                           current_context + user_input, temp=0.7)
        # current_context has everything:
        # profile, risk%, charges, RAG evidence, recommendation

    else:
        # Extract features from description
        features = groq.chat(EXTRACTION_PROMPT, user_input, temp=0.3)
        # Returns: {"tenure":1, "MonthlyCharges":95, "Contract":0...}

        # XGBoost prediction (local, no internet, 0.01s)
        prob = model.predict_proba([features])[0][1]

        # RAG search (local, no internet, 0.05s)
        similar = collection.query(
            query_texts=[customer_to_text(features)],
            n_results=5
        )
        # Returns 5 most similar past customers + outcomes

        # Recommendation (Groq API, ~1s)
        recommendation = groq.chat(
            RECOMMENDATION_PROMPT,
            f"Risk:{prob:.1%}, RAG:{similar_summary}", temp=0.5
        )

        # Store context for follow-ups
        current_context = f"""
Profile: {customer_to_text(features)}
Risk: {prob:.1%}
Monthly: ${features['MonthlyCharges']}
Annual value: ${features['MonthlyCharges'] * 12}
Revenue lost if churns: ${features['MonthlyCharges'] * remaining_months}
Contract reduces risk to: {contract_prob:.1%}
RAG: {similar_summary}
Recommendation: {recommendation}
        """

# LIVE RAG UPDATE — grows database after each interaction
def add_to_rag(features, outcome, intervention):
    collection.add(
        documents=[customer_to_text(features)],
        metadatas=[{"outcome": outcome, "intervention": intervention}],
        ids=[f"new_{timestamp}"]
    )
    # RAG database grows by 1 after every recorded outcome
    # More interactions = better future recommendations
```

---

## The RAG Database — `chroma_db/` Explained

```
data_level0.bin:
  The actual 1,409 vectors
  Each customer = 384 floating point numbers
  Binary format — ChromaDB manages this
  You never touch this directly

chroma.sqlite3:
  Human readable metadata
  Open with any SQLite browser
  Contains: text, outcome, intervention, risk for each customer
  SELECT * FROM embeddings → see all 1,409 records

link_lists.bin (HNSW graph):
  How vectors are connected for fast search
  HNSW = Hierarchical Navigable Small World algorithm
  Brute force: compare to all 1,409 = 1,409 operations
  HNSW: follow graph connections = ~20 operations
  At 10 million customers: still ~20 operations
  This is why vector search is fast at scale
```

---

## Key Numbers

| Metric | Value | Meaning |
|--------|-------|---------|
| Dataset | 7,043 customers | Enough for reliable patterns |
| Features | 19 | Inputs to XGBoost |
| Training set | 5,634 (80%) | XGBoost learned from these |
| Test set = RAG | 1,409 (20%) | Evidence database |
| Accuracy | 80% | Solid for production |
| AUC | 0.841 | Near production-ready |
| Churn rate | 26.5% | Healthy class balance |
| Top driver | Contract (38.4%) | Single biggest predictor |
| High risk savings | ~$1,402 | ROI of one phone call |
| False negatives | 170 customers | $130,560 missed revenue |

---

## How to Run

```bash
git clone https://github.com/vipuljain675-projects/Churn-AI-predictor.git
cd Churn-AI-predictor
pip3 install pandas numpy scikit-learn xgboost shap chromadb groq joblib matplotlib seaborn
echo "GROQ_API_KEY=your_key_here" > .env
python3 notebooks/02_model.py      # train model (3 seconds)
python3 src/build_rag_db.py        # build RAG (30 seconds)
python3 src/chat_with_rag.py       # start ChurnAI
```

---

## Tech Stack

| Technology | Purpose | Why |
|------------|---------|-----|
| pandas | Load + clean data | Industry standard |
| scikit-learn | Encoding, splitting, metrics | Standard ML toolkit |
| XGBoost | Churn prediction | Best for tabular data |
| SHAP | Explain predictions | Mathematical, not guesswork |
| ChromaDB | Vector RAG database | Free, local, no API key |
| Groq API | LLaMA 70B | Free tier, fastest inference |
| joblib | Save/load model | Standard serialization |
| matplotlib/seaborn | Charts | Standard visualisation |

---

## Interview Answers

**30-second pitch:**
"ChurnAI predicts which telecom customers will cancel before they do. XGBoost predicts churn probability, SHAP explains why, ChromaDB RAG finds evidence from 1,409 past customers, and LLaMA 70B gives actionable recommendations through a conversational interface with full memory."

**Scale to millions?**
"Events stream via Kafka to Flink which embeds profiles and upserts to ChromaDB in real time. XGBoost retrains monthly. RAG grows continuously with every recorded outcome."

**Why XGBoost not neural network?**
"For 7,000 rows of tabular data, XGBoost consistently outperforms neural networks. Neural networks need millions of examples. XGBoost trains in 3 seconds and wins most Kaggle competitions for this exact problem type."

**Business value?**
"Each saved high-risk customer = $1,402 retained revenue. 170 missed churners in our test = $130,560 lost. At company scale, even 5% improvement in detection = millions saved annually."

---

## What I Learned

1. ML is pattern recognition — not magic
2. Data cleaning matters more than the algorithm — TotalCharges was broken
3. 80% accuracy sounds good — 170 missed churners = $130K reality check
4. RAG eliminates hallucination — evidence beats generic LLM advice
5. Architecture beats model size — small well-designed > large poorly-designed
6. LLM is just the voice — XGBoost and ChromaDB do the actual thinking
7. Conversation memory requires explicit design — routers and context storage

---

*Built in one day. Vibe coded with Claude. But understood every line.* 🇮🇳