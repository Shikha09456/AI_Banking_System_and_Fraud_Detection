# 🏦 AI Banking & Fraud Detection System

An end-to-end **AI-powered banking customer support and fraud detection system** that combines **Machine Learning, Retrieval-Augmented Generation (RAG), semantic search, and a local LLM** to analyze customer banking queries and generate context-aware responses.

The system is designed to demonstrate how a modern banking AI pipeline can classify customer intent, assess risk, detect sentiment, retrieve relevant banking policies/documents, and generate a final response using a **free local LLM through Ollama**.

---

## 📌 Project Overview

Traditional banking support systems often rely on fixed rules, keyword matching, or manual investigation. This project combines multiple AI components into a single pipeline.

A customer enters a natural-language banking query such as:

> **"I see a transaction of ₹10,000 I didn't make."**

The system processes the query through:

1. **Intent Classification**
2. **Sentiment Analysis**
3. **Risk Classification**
4. **Category Mapping**
5. **RAG-based Document Retrieval**
6. **Local LLM Response Generation**
7. **Recommended Next Action**

### Example

**Input**

```text
I see a transaction of ₹10,000 I didn't make
```

**Possible system output**

```text
Intent: Fraud / Unauthorized Transaction
Sentiment: Negative
Risk Level: HIGH
RAG Category: fraud

Recommended Action:
Block/restrict the card or transaction and escalate
to the fraud investigation team.
```

The final customer-facing response is generated using relevant banking knowledge retrieved from the project's document repository.

---

# 🚀 Key Features

## 1. Customer Intent Classification

The system predicts the intent behind a customer's banking query.

Example categories include:

- Fraud / Unauthorized Transaction
- Refund
- Dispute
- KYC
- Loan
- General Banking Query

The project supports a separately trained:

```text
banking_intent_model.pkl
banking_intent_tfidf.pkl
```

---

## 2. Risk Classification

The risk model predicts the risk associated with the customer's query.

Supported normalized risk levels:

- 🟢 LOW
- 🟡 MEDIUM
- 🔴 HIGH

For example, an unauthorized transaction can be classified as high risk and routed toward fraud escalation.

Files:

```text
risk_category_model.pkl
risk_category_tfidf.pkl
```

---

## 3. Sentiment Analysis

The sentiment model analyzes the customer's emotional tone.

Example outputs:

- Positive
- Neutral
- Negative

Files:

```text
sentiment_model.pkl
sentiment_tfidf.pkl
```

Sentiment information is passed to the downstream LLM to help generate an appropriate customer response.

---

## 4. Retrieval-Augmented Generation (RAG)

The application uses RAG to ground LLM responses in banking documents.

Supported document formats:

```text
.txt
.md
.json
.pdf
```

The pipeline:

```text
Banking Documents
       ↓
Text Extraction
       ↓
Chunking
       ↓
Sentence Transformer Embeddings
       ↓
FAISS Vector Index
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
Local LLM
```

This reduces the need for the LLM to rely only on its general knowledge.

---

## 5. Category-Aware Retrieval

Documents are automatically categorized into areas such as:

```text
fraud
refund
kyc
loan
general
```

The predicted intent is mapped to one of these categories.

For example:

```text
Intent:
Unauthorized Transaction

        ↓

RAG Category:
fraud

        ↓

Retrieve:
Fraud-related banking documents
```

If category-specific documents are not available, the system falls back to semantic retrieval.

---

## 6. Local LLM with Ollama

The project uses a local LLM instead of a paid cloud API.

Default model:

```text
gemma3:1b
```

The LLM is accessed through:

```text
Ollama
```

This means the application can generate responses locally without requiring an OpenAI API key.

---

## 7. Free Embedding Model

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is used to convert documents and customer queries into vector representations for semantic search.

---

## 8. FAISS Vector Search

The project uses **FAISS** for efficient similarity search.

Embeddings are normalized and stored in a FAISS inner-product index.

Conceptually:

```text
Customer Query
      ↓
Embedding
      ↓
FAISS Similarity Search
      ↓
Top Relevant Chunks
```

---

## 9. Recommended Next Action

The application provides a rule-based recommended action based on:

```text
Intent Category + Risk Level
```

Example:

```text
Fraud + HIGH
        ↓
Block/restrict card or transaction
        ↓
Escalate to fraud investigation team
```

This provides an actionable output in addition to the generated customer response.

---

## 10. Streamlit Interface

The complete application is available through a Streamlit web interface.

The interface displays:

- Customer query input
- Intent
- Sentiment
- Risk level
- RAG category
- AI-generated response
- Recommended next action
- Retrieved documents
- System/model status

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   CUSTOMER QUERY    │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   Streamlit Interface  │
                  └──────────┬─────────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │Intent Model │ │Risk Model   │ │Sentiment    │
      │+ TF-IDF     │ │+ TF-IDF     │ │Model+TF-IDF │
      └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Category Mapping│
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    RAG Retriever    │
                  └──────────┬──────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
      ┌──────────────┐               ┌──────────────┐
      │ Banking Docs │               │ FAISS Index  │
      │ PDF/JSON/TXT │──────────────►│ + Embeddings │
      └──────────────┘               └──────┬───────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Relevant Chunks  │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ Ollama + Gemma   │
                                   │ Local LLM        │
                                   └────────┬─────────┘
                                            │
                                            ▼
                            ┌──────────────────────────┐
                            │ Final Customer Response  │
                            │ + Recommended Action    │
                            └──────────────────────────┘
```

---

# 🔄 End-to-End Workflow

```text
1. Customer enters query
            ↓
2. Intent prediction
            ↓
3. Risk prediction
            ↓
4. Sentiment prediction
            ↓
5. Intent mapped to RAG category
            ↓
6. Banking documents loaded
            ↓
7. Documents split into chunks
            ↓
8. Chunks converted into embeddings
            ↓
9. FAISS performs semantic retrieval
            ↓
10. Relevant context sent to Gemma
            ↓
11. Gemma generates customer response
            ↓
12. System displays recommended action
```

---

# 📁 Project Structure

```text
AI_Banking_system/
│
├── app.py
├── final_app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── models/
│   ├── banking_intent_model.pkl
│   ├── banking_intent_tfidf.pkl
│   ├── risk_category_model.pkl
│   ├── risk_category_tfidf.pkl
│   ├── sentiment_model.pkl
│   └── sentiment_tfidf.pkl
│
└── Data/
    ├── banking policies
    ├── fraud documents
    ├── refund documents
    ├── KYC documents
    ├── loan documents
    └── other banking knowledge
```

> Keep the actual dataset/document structure consistent with your project. The application recursively reads supported files inside the `Data/` directory.

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| ML | Scikit-learn |
| Text Vectorization | TF-IDF |
| Model Serialization | Joblib / Pickle |
| Embeddings | Sentence Transformers |
| Vector Database/Search | FAISS |
| LLM Runtime | Ollama |
| Local LLM | Gemma 3 1B |
| PDF Processing | pypdf |
| Configuration | python-dotenv |
| Numerical Computing | NumPy |

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd AI_Banking_system
```

Replace `<YOUR_USERNAME>` and `<YOUR_REPOSITORY>` with your GitHub details.

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the main packages are:

```bash
pip install streamlit
pip install scikit-learn
pip install joblib
pip install numpy
pip install pandas
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
pip install python-dotenv
pip install ollama
```

---

# 🤖 Ollama Setup

Install Ollama on your system and make sure the Ollama service is running.

Then pull the Gemma model:

```bash
ollama pull gemma3:1b
```

Verify that the model is available:

```bash
ollama list
```

You should see something similar to:

```text
gemma3:1b
```

The application expects Ollama to be available at:

```text
http://localhost:11434
```

---

# ⚙️ Environment Variables

Create a `.env` file in the project root.

Example:

```env
OLLAMA_MODEL=gemma3:1b
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The application loads these values automatically.

---

# 🧠 ML Model Files

The application expects six trained artifacts.

```text
models/
├── banking_intent_model.pkl
├── banking_intent_tfidf.pkl
├── risk_category_model.pkl
├── risk_category_tfidf.pkl
├── sentiment_model.pkl
└── sentiment_tfidf.pkl
```

The models and their TF-IDF vectorizers should be saved from the **same training pipeline**.

Example:

```python
import joblib

joblib.dump(intent_model, "models/banking_intent_model.pkl")
joblib.dump(intent_tfidf, "models/banking_intent_tfidf.pkl")

joblib.dump(risk_model, "models/risk_category_model.pkl")
joblib.dump(risk_tfidf, "models/risk_category_tfidf.pkl")

joblib.dump(sentiment_model, "models/sentiment_model.pkl")
joblib.dump(sentiment_tfidf, "models/sentiment_tfidf.pkl")
```

---

# ▶️ Running the Application

From the project root:

```bash
streamlit run final_app.py
```

Or if your main application is named `app.py`:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open that URL in your browser.

---

# 🧪 Example Queries

## Fraud

```text
I see a transaction of ₹10,000 I didn't make
```

Expected behavior:

```text
Intent → Fraud / Unauthorized Transaction
Risk → HIGH
Category → fraud
Action → Block/restrict and escalate
```

---

## Refund

```text
I requested a refund but haven't received it yet
```

Expected behavior:

```text
Intent → Refund
Category → refund
Action → Verify refund/dispute details
```

---

## KYC

```text
Why do I need to complete KYC?
```

Expected behavior:

```text
Intent → KYC
Category → kyc
Action → Follow KYC verification policy
```

---

## Loan

```text
What documents are required for my loan application?
```

Expected behavior:

```text
Intent → Loan
Category → loan
Action → Verify loan details and follow loan policy
```

---

# 🔐 Model Loading Design

The application uses a robust artifact-loading strategy.

It first attempts:

```python
joblib.load()
```

and then falls back to:

```python
pickle.load()
```

The application also constructs model paths relative to `app.py`:

```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
```

This avoids common errors caused by Streamlit being launched from a different working directory.

---

# 📚 RAG Implementation Details

## Document ingestion

The application recursively scans:

```text
Data/
```

and supports:

```text
.txt
.md
.json
.pdf
```

---

## Chunking

Documents are split into overlapping text chunks.

Default settings:

```text
Chunk size: 450 words
Overlap: 80 words
```

Overlap helps preserve context between adjacent chunks.

---

## Embedding

Each chunk is converted into a vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The customer query is embedded using the same model.

---

## Similarity Search

FAISS uses an inner-product index:

```text
FAISS IndexFlatIP
```

Embeddings are normalized, making the similarity search behave similarly to cosine similarity.

---

# 🧩 Why RAG?

A generic LLM may not know the organization's latest or project-specific banking policies.

RAG allows the system to provide relevant internal knowledge to the LLM:

```text
Customer Query
      ↓
Retrieve Relevant Banking Policy
      ↓
Provide Policy as Context
      ↓
Generate Grounded Response
```

This helps reduce unsupported responses and keeps the generated answer aligned with the available project documents.

---

# 🛡️ Fraud Detection Flow

For an unauthorized transaction:

```text
Customer:
"I see a transaction of ₹10,000 I didn't make"
              ↓
       Intent Model
              ↓
    Unauthorized Transaction
              ↓
        Category = fraud
              ↓
         Risk Model
              ↓
          HIGH RISK
              ↓
      Fraud Documents
              ↓
       Relevant Context
              ↓
        Gemma Local LLM
              ↓
 Customer Response + Action
              ↓
Block/Restrict + Fraud Escalation
```

---

# 🎯 Business Use Cases

This architecture can be adapted to several banking workflows:

### Fraud Detection

- Unauthorized transactions
- Suspicious activity
- Card fraud
- Transaction disputes

### Customer Support

- Account questions
- Transaction questions
- Banking policies
- General support

### KYC

- KYC requirements
- Verification questions
- Document requirements

### Loans

- Loan eligibility
- Loan documentation
- Loan processing questions

### Refunds & Disputes

- Refund status
- Chargebacks
- Transaction disputes

---

# 📊 Model Evaluation

Before deploying trained models, evaluate each classifier using:

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    classification_report(
        y_test,
        y_pred,
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)
```

For fraud-related applications, **recall and precision should be considered alongside accuracy**, because false negatives can be particularly important.

---

# ⚡ Performance Considerations

The application uses Streamlit caching for expensive resources such as:

- ML models
- Embedding model
- FAISS knowledge base
- Ollama client

This avoids repeatedly loading large models during Streamlit reruns.

For larger document collections, possible improvements include:

- Persistent FAISS indexes
- Smaller chunk sizes
- Better metadata filtering
- Hybrid keyword + semantic search
- Reranking
- Batch embedding
- Async LLM requests
- Quantized local models

---

# 🔒 Security Considerations

This project is intended as an educational/prototype banking AI system.

For production deployment:

- Never expose customer PII unnecessarily.
- Encrypt sensitive data.
- Apply authentication and authorization.
- Use secure secrets management.
- Log access securely.
- Validate model inputs.
- Implement human review for high-risk fraud cases.
- Do not allow the LLM to independently perform irreversible banking actions.
- Keep banking policies version-controlled.
- Audit model predictions and generated responses.

**The recommended actions displayed by this project should not be treated as automatic authorization to block accounts, transfer funds, or take other irreversible actions.**

---

# ⚠️ Limitations

This project is a prototype and has several limitations:

1. Model performance depends on the quality and coverage of the training dataset.
2. RAG quality depends on the documents stored in `Data/`.
3. Automatically inferred document categories may not always be correct.
4. Local LLM responses depend on the selected Ollama model.
5. The system does not connect directly to a real banking transaction system.
6. The recommended action is not a replacement for bank fraud policies or human investigation.
7. Model artifacts must be compatible with the Python/scikit-learn environment used to load them.

---

# 🔮 Future Improvements

Potential extensions include:

- Real-time transaction monitoring
- Anomaly detection using transaction behavior
- XGBoost/LightGBM fraud scoring
- Neural-network-based intent classification
- Hybrid RAG
- RAG reranking
- Persistent vector database
- Conversation memory
- Multilingual banking support
- Voice-based banking assistant
- Human-in-the-loop fraud escalation
- Explainable AI for risk predictions
- Model monitoring and drift detection
- Feedback-based retraining
- Production API using FastAPI
- Docker deployment
- Cloud deployment
- Role-based banking staff dashboard

---

# 🧪 Suggested Production Architecture

A production version could evolve into:

```text
                    Customer
                       │
                       ▼
                API / Web App
                       │
                       ▼
              Query Processing
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Intent         Risk       Sentiment
        Model         Model        Model
          │            │            │
          └────────────┼────────────┘
                       ▼
                 RAG Retriever
                       │
                       ▼
              Vector Database
                       │
                       ▼
              Policy / Knowledge
                       │
                       ▼
                  LLM Layer
                       │
                       ▼
              Response Generator
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Customer Reply       Human Review
                             for High Risk
```

---

# 🛠️ Troubleshooting

## Model not found

If you see:

```text
FileNotFoundError:
models/banking_intent_model.pkl
```

verify that the structure is:

```text
AI_Banking_system/
├── final_app.py
└── models/
    └── banking_intent_model.pkl
```

---

## Model loading error

If you see errors such as:

```text
STACK_GLOBAL requires str
```

or:

```text
invalid load key
```

the issue may be the serialization/contents of the `.pkl` artifact rather than the file path.

Regenerate the artifact using the same environment and:

```python
joblib.dump(model, "models/model.pkl")
```

---

## Ollama error

If the application says the local LLM cannot be reached:

```bash
ollama list
```

Then verify that:

```text
gemma3:1b
```

exists.

If necessary:

```bash
ollama pull gemma3:1b
```

---

## No RAG documents

Verify that documents are inside:

```text
Data/
```

and use one of:

```text
.txt
.md
.json
.pdf
```

---

# 📜 License

Add your preferred license before publishing.

For example, if you choose the MIT License:

```text
MIT License
```

You can create a `LICENSE` file in the repository and include the complete license text.

---

# 👩‍💻 Author

**Shikha Kumari**

AI / Machine Learning Project

---

# ⭐ Project Highlights

This project demonstrates an end-to-end **Generative AI + Machine Learning + RAG** architecture for banking applications.

### Core AI Components

```text
Machine Learning
      +
TF-IDF
      +
Sentiment Analysis
      +
Risk Classification
      +
Semantic Embeddings
      +
FAISS
      +
RAG
      +
Ollama
      +
Gemma
      +
Streamlit
```

### Complete Pipeline

```text
Customer Query
      ↓
Intent Detection
      ↓
Risk Assessment
      ↓
Sentiment Analysis
      ↓
Category Mapping
      ↓
RAG Retrieval
      ↓
Banking Knowledge
      ↓
Gemma Local LLM
      ↓
Customer Response
      ↓
Recommended Action
```

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and using the architecture as a starting point for your own AI-powered financial applications.
