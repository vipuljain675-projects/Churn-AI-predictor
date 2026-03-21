# ChurnAI — Customer Retention Intelligence System

> Built by Vipul Jain | 2nd Semester | Built in 1 day

A production-grade AI system that predicts customer churn, explains WHY using SHAP, retrieves evidence from 1,409 past customers using RAG, and gives evidence-based retention recommendations through a conversational interface.

---

## What This Project Does

```
You describe a customer in plain English
        ↓
XGBoost predicts: 78.5% chance of churning
        ↓
ChromaDB finds 5 most similar past customers
        ↓
LLaMA 70B combines prediction + evidence
        ↓
"Call within 24 hours. Offer annual contract.
 Discounts failed twice for this profile.
 Revenue at stake: $1,402"
```

You can then ask follow-up questions:
- "What if he leaves anyway?"
- "Should we spend money retaining him?"
- "What if 100 customers like him leave tomorrow?"

The system remembers context across the entire conversation.

---

## Demo — What It Looks Like

```
You → Customer, 1 month old, $95/month, month-to-month, fiber optic, no security

ChurnAI →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 XGBOOST PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result:      🔴 HIGH RISK
Probability: 81.9% chance of churning
Contract upgrade → 35.7% | Saves $1402.32

🔍 RAG EVIDENCE — 5 Most Similar Past Customers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Customer 1: ❌ CHURNED | Tried: discount → FAILED
Customer 2: ✅ STAYED  | Tried: annual contract → WORKED
Customer 3: ❌ CHURNED | Tried: no action → FAILED

🤖 CHURNAI RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT'S HAPPENING: Customer is at 81.9% churn risk due to 
month-to-month contract and no added services.

WHAT THE DATA SHOWS: Discounts failed twice for this profile. 
Annual contract is the only proven intervention.

WHAT YOU MUST DO RIGHT NOW:
1. Call within 24 hours
2. Offer annual contract with 15% first-month discount  
3. If refused: add free security bundle for 90 days

BOTTOM LINE: Call today. Annual contract only. $1,402 at stake.

You → What if 100 customers like him leave tomorrow?

ChurnAI → Revenue loss = 100 × $95 × 12 = $114,000/year.
Competitor propaganda requires immediate response strategy...
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER (chat)                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              ROUTER (LLaMA 70B)                     │
│   Is this a NEW customer or FOLLOW-UP question?     │
└──────────┬──────────────────────┬───────────────────┘
           │ NEW                  │ FOLLOW-UP
           ▼                      ▼
┌──────────────────┐   ┌─────────────────────────────┐
│ FEATURE          │   │ ANSWER using stored context  │
│ EXTRACTION       │   │ No new prediction needed     │
│ (LLaMA 70B)      │   └─────────────────────────────┘
└──────────┬───────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              TWO PARALLEL OPERATIONS                  │
│                                                       │
│  XGBoost Model          ChromaDB RAG                 │
│  ─────────────          ─────────────                │
│  Predicts churn %       Finds 5 similar              │
│  from features          past customers               │
│  (0.01 seconds)         (0.05 seconds)               │
└──────────┬──────────────────────┬────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────────────────────────────────────────┐
│           RECOMMENDATION ENGINE (LLaMA 70B)          │
│   Combines: prediction + RAG evidence                │
│   Outputs: specific actions with business math       │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
churn-predictor/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← 7,043 real telecom customers
├── models/
│   ├── churn_model.pkl                         ← trained XGBoost model
│   └── feature_names.pkl                       ← list of feature names
├── notebooks/
│   ├── 01_explore.py                           ← Stage 1: understand the data
│   ├── 02_model.py                             ← Stage 2: train XGBoost
│   └── 03_shap.py                              ← Stage 3: explain predictions
├── src/
│   ├── chat.py                                 ← basic chat (no RAG)
│   ├── build_rag_db.py                         ← builds ChromaDB (run once)
│   └── chat_with_rag.py                        ← full system with RAG
└── chroma_db/                                  ← vector database (auto-created)
```

---

# ML Education — Every Concept Explained

This section explains every ML concept used in this project from scratch.

---

## 1. What is Machine Learning?

Normal programming: you write rules.
```python
if tenure < 3 and contract == "month-to-month":
    return "will churn"
```
Problem: you can't write rules for every combination of 19 features.

Machine Learning: you show the computer examples and it finds the rules itself.
```python
# You give it 7,043 examples
# It finds: "month-to-month + new + high bill + no security = 73% churn"
# You didn't write this rule. It discovered it.
model.fit(X_train, y_train)
```

---

## 2. What is a Dataset?

Our dataset has 7,043 rows (customers) and 21 columns (features).

```
customerID | tenure | MonthlyCharges | Contract    | Churn
7590-VHVEG |   2    |     29.85      | Month-month |  No
5575-GNVDE |  34    |     56.95      | Two year    |  No
3668-QPYBK |   2    |     53.85      | Month-month |  Yes
```

- **Features (X)** = inputs = everything except Churn column
- **Target (y)** = output = the Churn column
- **Goal** = learn the relationship between X and y

---

## 3. Library: pandas

```python
import pandas as pd
```

**What it is:** A library for working with tables of data. Think Excel but in Python.

**Why we used it:**
```python
# Load CSV into a DataFrame (table)
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix a column with bad data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop a column we don't need
df = df.drop('customerID', axis=1)

# Convert Yes/No to 1/0
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Get all text columns
df.select_dtypes(include=['object']).columns
```

**Real life:** Every data scientist uses pandas daily. It's the most important library in data science.

---

## 4. Library: numpy

```python
import numpy as np
```

**What it is:** A library for mathematical operations on arrays of numbers.

**Why we used it:**
```python
# Create arrays of zeros
np.zeros(10)

# Mathematical operations on entire arrays at once
# Instead of: for each number, multiply by 2
# Just: array * 2
```

**Real life:** pandas is built on top of numpy. Every ML library uses numpy internally.

---

## 5. What is Train/Test Split?

This is THE most important concept in ML.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why:** If you test on the same data you trained on, the model just memorizes answers. That's cheating — it won't work on new customers.

```
7,043 customers
├── 5,634 (80%) → model LEARNS from these (training set)
└── 1,409 (20%) → model TESTED on these (test set)
                   model has NEVER seen these before
```

**Analogy:** 
- Training set = textbook you study from
- Test set = exam questions you've never seen
- Good score on exam = model actually learned, didn't memorize

**random_state=42:** Makes the split identical every time you run the code. Without it, you'd get a different split each run and different results. 42 is a convention (from Hitchhiker's Guide to the Galaxy 😄).

**stratify=y:** Ensures the 26.5% churn ratio is preserved in BOTH splits. Without it, by random chance you might get 40% churners in training and 10% in test — unfair comparison.

---

## 6. What is Label Encoding?

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
# Month-to-month → 0
# One year       → 1
# Two year       → 2
```

**Why:** ML models only understand numbers. "Month-to-month" means nothing to XGBoost. 0, 1, 2 does.

**What fit_transform does:**
- `fit` = learns the mapping (Month-to-month=0, One year=1, Two year=2)
- `transform` = applies the mapping to convert text to numbers

---

## 7. What is XGBoost?

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,    # build 200 decision trees
    max_depth=4,         # each tree max 4 levels deep
    learning_rate=0.1,   # how much each tree corrects previous
    subsample=0.8,       # use 80% of data per tree
    colsample_bytree=0.8 # use 80% of features per tree
)
model.fit(X_train, y_train)
```

**What it is:** XGBoost = Extreme Gradient Boosting. It builds many small decision trees and combines them.

**How it works:**

```
Tree 1: learns basic patterns
  → makes mistakes on some customers

Tree 2: focuses on the mistakes Tree 1 made
  → corrects them, makes new mistakes

Tree 3: focuses on mistakes Tree 1+2 made
  → corrects them...

...200 trees later...

Final answer = weighted combination of all 200 trees
```

This is called **ensemble learning** — wisdom of crowds. 200 weak trees together are stronger than one perfect tree.

**Why XGBoost specifically:**
- Wins most Kaggle competitions for tabular data
- Handles missing values automatically
- Very fast
- Works well with 7,000 rows

**Parameters explained:**
- `n_estimators=200` → build 200 trees (more = better but slower)
- `max_depth=4` → each tree can ask max 4 questions (prevents memorizing)
- `learning_rate=0.1` → each tree contributes 10% of correction (slow and steady)
- `subsample=0.8` → each tree sees 80% of rows randomly (prevents overfitting)
- `colsample_bytree=0.8` → each tree sees 80% of features randomly (prevents overfitting)

---

## 8. What is Overfitting?

```
Overfitting = model memorized training data instead of learning patterns

Example:
Training accuracy: 99%  ← memorized every customer
Test accuracy:     60%  ← fails on new customers

Good model:
Training accuracy: 83%  ← learned real patterns
Test accuracy:     80%  ← works on new customers too
```

Our `subsample` and `colsample_bytree` parameters prevent this by forcing each tree to learn from random subsets — can't memorize if it never sees the full data.

---

## 9. What is Accuracy vs AUC?

```python
accuracy = accuracy_score(y_test, y_pred)  # 80%
auc = roc_auc_score(y_test, y_pred_proba)  # 0.841
```

**Accuracy (80%):** "The model correctly classified 80% of customers."

Problem with accuracy: if 95% of customers stay, a model that ALWAYS says "will stay" gets 95% accuracy. It's useless but looks great.

**AUC (0.841):** "84.1% of the time, the model correctly ranks a churner higher than a non-churner."

AUC doesn't care about the threshold. It measures how well the model SEPARATES the two classes. AUC > 0.85 = production ready. Ours is 0.841 — very close.

---

## 10. What is a Confusion Matrix?

```
                  Predicted Stay  Predicted Churn
Actually Stayed       923              112
Actually Churned      170              204
```

- **923 (True Negative):** Said "will stay" → actually stayed ✅
- **204 (True Positive):** Said "will churn" → actually churned ✅
- **112 (False Positive):** Said "will churn" → actually stayed ❌ (wasted retention offer)
- **170 (False Negative):** Said "will stay" → actually churned ❌ (missed churner = lost revenue)

Business cost of 170 missed churners:
```
170 customers × $64/month × 12 months = $130,560 lost revenue
```

---

## 11. Library: joblib

```python
import joblib

# Save trained model
joblib.dump(model, 'models/churn_model.pkl')

# Load it later without retraining
model = joblib.load('models/churn_model.pkl')
```

**What it is:** Saves Python objects to disk. Without this, you'd have to retrain the model every time you start the program (3 seconds wasted each time).

**Why .pkl:** Pickle format — Python's way of serializing objects.

---

## 12. What is SHAP?

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**Problem:** XGBoost says "78% churn" but doesn't explain WHY.

**SHAP solves this:** SHapley Additive exPlanations — mathematical framework that assigns each feature a "blame" score for the prediction.

```
Customer prediction: 78% churn
SHAP breakdown:
  tenure = 1 month    → +1.017 churn risk  (biggest factor)
  Contract = 0        → +0.777 churn risk
  No security         → +0.294 churn risk
  High charges        → +0.156 churn risk
  Has phone service   → -0.089 churn risk (reduces risk slightly)
```

Now you know EXACTLY why this customer is predicted to churn. Not just a number — a reason.

**Why it matters for business:** A retention manager can act on specific reasons. "Offer annual contract" targets the Contract SHAP value directly.

---

## 13. What is RAG?

RAG = Retrieval Augmented Generation

**Problem:** LLaMA gives generic advice. "Offer a discount" is generic. What actually worked for similar customers?

**RAG solution:** Build a database of past customers with known outcomes. When a new customer comes in, find similar past customers and show the LLM what worked for them.

```
New customer → convert to vector → search ChromaDB
                                          ↓
                            Returns 5 most similar past customers
                            with their actual outcomes
                                          ↓
                            LLM gets evidence-based context
                                          ↓
                            "Annual contract worked 3/3 times
                             for this exact profile. Discounts
                             failed twice. Offer annual contract."
```

---

## 14. What is a Vector / Embedding?

```python
# Customer in numbers
customer = {"tenure": 2, "MonthlyCharges": 90, "Contract": 0...}

# Customer as text
text = "Customer with 2 months tenure, paying 90 dollars monthly,
        month-to-month contract, no security..."

# Customer as vector (embedding)
vector = [0.23, 0.87, 0.12, 0.94, 0.33...]  # 384 numbers
```

**Why convert to vector?** So we can measure SIMILARITY mathematically.

```
Customer A vector: [0.23, 0.87, 0.12...]
Customer B vector: [0.24, 0.85, 0.13...]
Distance = very small → customers are SIMILAR

Customer A vector: [0.23, 0.87, 0.12...]
Customer C vector: [0.91, 0.12, 0.78...]
Distance = very large → customers are DIFFERENT
```

ChromaDB stores these vectors and finds the closest ones in milliseconds.

---

## 15. Library: chromadb

```python
import chromadb

# Create persistent database (saves to disk)
client = chromadb.PersistentClient(path="chroma_db")

# Create a collection (like a table)
collection = client.create_collection("customers")

# Store a customer
collection.add(
    documents=["Customer with 2 months tenure, paying 90 dollars..."],
    metadatas=[{"outcome": "CHURNED", "intervention": "discount_failed"}],
    ids=["customer_1"]
)

# Find similar customers
results = collection.query(
    query_texts=["new customer description"],
    n_results=5
)
```

**What it does:**
1. Converts text to vectors automatically (uses built-in embedding model)
2. Stores vectors + metadata efficiently
3. Finds K most similar vectors using cosine similarity

**Why ChromaDB:** Free, local, no API key, easy to use, persists to disk.

---

## 16. What is Cosine Similarity?

ChromaDB uses this to find similar customers.

```
Two vectors are similar if they point in the same direction.

Customer A: [0.23, 0.87, 0.12]  ←─ these point in
Customer B: [0.24, 0.85, 0.13]  ←─ nearly same direction
Cosine similarity = 0.999 (very similar)

Customer A: [0.23, 0.87, 0.12]
Customer C: [0.91, 0.12, 0.78]  ← different direction
Cosine similarity = 0.412 (not similar)
```

In our output: "86.4% similar" = cosine similarity of 0.864.

---

## 17. What is an LLM?

LLM = Large Language Model (LLaMA, GPT, Claude)

**What it is:** A neural network trained on trillions of words from the internet. It learned to predict the next word — and in doing so, it learned language, reasoning, math, coding, everything.

**How we use it:**
```python
from groq import Groq

client = Groq(api_key="...")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # 70 billion parameter model
    messages=[
        {"role": "system", "content": "You are ChurnAI..."},
        {"role": "user", "content": "Customer, 2 months, $90/month..."}
    ],
    max_tokens=512,   # max length of response
    temperature=0.7,  # 0=deterministic, 1=creative
)
answer = response.choices[0].message.content
```

**temperature explained:**
- `0.0` = always same answer (good for routing decisions)
- `0.3` = mostly consistent (good for feature extraction)
- `0.7` = somewhat creative (good for recommendations)
- `1.0` = very random (good for creative writing)

---

## 18. What is Fine-tuning? (Used in Airavat, not ChurnAI)

**Base LLaMA:** Knows everything generally but doesn't know VAJRA format, India doctrines, causal chain reasoning.

**Fine-tuning:** Show it 700 examples of exactly the style you want → model adjusts its weights → now it always responds in that style.

```
Before fine-tuning:
Q: "Why is USA encircling India?"
A: "The United States has various foreign policy interests..."
(generic, boring, no doctrine names)

After fine-tuning on 700 VAJRA examples:
Q: "Why is USA encircling India?"
A: "DIRECT ANSWER: This is the 3.5 FRONT ENCIRCLEMENT doctrine...
    THE PATTERN: PROXY BALANCING activated...
    INDIA'S MOVE: Accelerate Project 75 Alpha SSN program..."
(specific, structured, India-first)
```

**Why we needed Colab:** Fine-tuning 1B model = adjusting 1 billion numbers = needs GPU.
**Why ChurnAI doesn't need it:** XGBoost trains in 3 seconds on CPU.

---

## 19. What is a System Prompt?

```python
SYSTEM_PROMPT = """You are ChurnAI — a customer retention analyst.
Your job is to predict churn and give specific retention advice..."""
```

**What it is:** Instructions given to the LLM before the conversation starts. Sets its personality, role, and rules.

Think of it as the LLM's job description. Without it, LLaMA is a general assistant. With it, it becomes ChurnAI.

**The three-prompt architecture we used:**
- `ROUTER_PROMPT` → decides NEW or FOLLOWUP (temperature=0.0, deterministic)
- `EXTRACTION_PROMPT` → extracts customer features (temperature=0.3, consistent)
- `RECOMMENDATION_PROMPT` → gives advice (temperature=0.5, balanced)
- `FOLLOWUP_PROMPT` → answers follow-ups (temperature=0.7, conversational)

Different temperatures for different jobs — this is production thinking.

---

## 20. What is an API?

```python
from groq import Groq
client = Groq(api_key="gsk_...")
```

**API = Application Programming Interface** = a way to use someone else's computer.

When we call Groq API:
1. Our code sends customer description to Groq's servers
2. Groq runs LLaMA 70B on their GPUs (which we can't afford)
3. They send back the response
4. We display it

We pay nothing (free tier). They run a 70B model requiring 4x A100 GPUs (worth ₹1 crore+).

**Why Groq specifically:** Groq built custom hardware (LPU chips) that runs LLaMA 3x faster than GPU. Free tier gives you access to LLaMA 70B — the same model Meta spent billions training.

---

## Key Numbers to Remember for Interviews

| Metric | Value | What It Means |
|--------|-------|---------------|
| Dataset size | 7,043 customers | Enough to train a reliable model |
| Features | 19 | Inputs to XGBoost |
| Train/Test split | 80/20 | Industry standard |
| Model accuracy | 80% | Solid for production |
| AUC score | 0.841 | Near production-ready threshold |
| Churn rate | 26.5% | Healthy class balance |
| RAG database | 1,409 customers | Past outcomes for evidence |
| Top churn driver | Contract type (38.4%) | Single biggest predictor |
| Revenue per customer | $64/month avg | Business context |
| Intervention savings | $1,402 (typical high risk) | ROI of retention action |

---

## How to Explain This Project in an Interview

**30-second version:**
"I built a customer churn prediction system that combines XGBoost for prediction, SHAP for explainability, and a ChromaDB RAG database of 1,409 past customer outcomes. The system provides evidence-based retention recommendations through a conversational interface that maintains memory across turns — including what-if analysis and business ROI calculations."

**The killer question answer:**
When they ask "how would you scale this to millions of customers?" say:
"Customer events from the web platform would stream via Kafka into a Flink processor which embeds customer profiles and upserts into ChromaDB in real time. The XGBoost model retrains monthly on accumulated data. The RAG database grows continuously with every customer interaction."

---

## How to Run This Project

```bash
# 1. Clone the repo
git clone https://github.com/vipuljain675-projects/Churn-AI-predictor.git
cd Churn-AI-predictor

# 2. Install dependencies
pip3 install pandas numpy scikit-learn xgboost shap chromadb groq joblib matplotlib seaborn

# 3. Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Train the model (takes 3 seconds)
python3 notebooks/02_model.py

# 5. Build RAG database (takes 30 seconds)
python3 src/build_rag_db.py

# 6. Start ChurnAI
python3 src/chat_with_rag.py
```

---

## Tech Stack

| Technology | Purpose | Why This One |
|------------|---------|--------------|
| Python | Everything | Industry standard for ML |
| pandas | Data manipulation | Most used data library |
| scikit-learn | Preprocessing, metrics | Standard ML toolkit |
| XGBoost | Churn prediction | Best for tabular data |
| SHAP | Explainability | Industry standard for model explanation |
| ChromaDB | Vector database (RAG) | Free, local, no API key |
| Groq API | LLM (LLaMA 70B) | Free tier, fastest inference |
| joblib | Save/load model | Standard Python serialization |
| matplotlib/seaborn | Charts | Standard visualization libraries |

---

## What I Learned Building This

1. **ML is not magic** — it's pattern recognition from examples
2. **Data cleaning matters more than the algorithm** — TotalCharges was broken, we fixed it
3. **80% accuracy sounds good but business context matters** — 170 missed churners = $130K lost
4. **RAG > hallucination** — evidence-based advice beats generic LLM advice
5. **Architecture > model size** — a small well-designed system beats a big poorly designed one
6. **The LLM is just the voice** — XGBoost and ChromaDB do the actual thinking
7. **Conversation memory is hard** — had to build a router to handle follow-ups properly

---

*Built in one day. Vibe coded with Claude. But understood every line.* 🇮🇳