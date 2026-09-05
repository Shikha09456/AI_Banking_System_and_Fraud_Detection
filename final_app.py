import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import ollama


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "Data"

load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ============================================================
# 2. MODEL PATHS
# ============================================================

INTENT_MODEL_PATH = MODEL_DIR / "banking_intent_model.pkl"
INTENT_TFIDF_PATH = MODEL_DIR / "banking_intent_tfidf.pkl"

RISK_MODEL_PATH = MODEL_DIR / "risk_category_model.pkl"
RISK_TFIDF_PATH = MODEL_DIR / "risk_category_tfidf.pkl"

SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
SENTIMENT_TFIDF_PATH = MODEL_DIR / "sentiment_tfidf.pkl"


# ============================================================
# 3. STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Banking & Fraud Detection",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 AI Banking & Fraud Detection System")
st.caption("ML Classification + RAG + Ollama/Gemma Local LLM")


# ============================================================
# 4. ROBUST MODEL / PICKLE LOADER
# ============================================================

def load_pickle(path: Path):
    """
    Load a saved ML artifact.

    First tries joblib because sklearn models/vectorizers are
    commonly saved using joblib.dump(). Then falls back to
    normal pickle.load().

    IMPORTANT:
    Path is always resolved from the folder containing app.py,
    so Streamlit's current working directory does not matter.
    """

    path = Path(path)

    if not path.exists():
        st.warning(f"⚠️ File not found: {path}")
        return None

    if not path.is_file():
        st.warning(f"⚠️ Path is not a file: {path}")
        return None

    # Attempt 1: joblib
    try:
        obj = joblib.load(path)
        return obj
    except Exception as joblib_error:
        joblib_message = str(joblib_error)

    # Attempt 2: normal pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception as pickle_error:
        st.warning(
            f"❌ Could not load {path.name}\n\n"
            f"Joblib error: {joblib_message}\n"
            f"Pickle error: {pickle_error}"
        )
        return None


# ============================================================
# 5. LABEL / PREDICTION HELPERS
# ============================================================

def normalize_label(value: Any) -> str:
    """Convert model prediction to a clean string."""

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return ""
        value = value[0]

    return str(value).strip()


def predict_with_model(
    model,
    vectorizer,
    text: str,
    default: str,
) -> str:
    """
    Supports two formats:

    1. Separate model + TF-IDF/vectorizer
       model.predict(vectorizer.transform([text]))

    2. A complete sklearn Pipeline saved as the model
       model.predict([text])
    """

    if model is None:
        return default

    try:
        # If model is a Pipeline, use raw text.
        if hasattr(model, "named_steps"):
            prediction = model.predict([text])
            return normalize_label(prediction)

        # Otherwise use the separately saved vectorizer.
        if vectorizer is None:
            st.warning(
                "⚠️ Vectorizer is missing. "
                f"Using default prediction: {default}"
            )
            return default

        X = vectorizer.transform([text])
        prediction = model.predict(X)

        return normalize_label(prediction)

    except Exception as exc:
        st.warning(
            f"⚠️ Prediction failed. Using '{default}'. "
            f"Details: {exc}"
        )
        return default


# ============================================================
# 6. CATEGORY NORMALIZATION
# ============================================================

def canonical_category(label: str) -> str:
    """
    Convert possible ML intent labels into the categories
    used by the RAG knowledge base.
    """

    x = (
        str(label)
        .lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
    )

    aliases = {
        "fraud": "fraud",
        "fraud detection": "fraud",
        "unauthorized transaction": "fraud",
        "unauthorised transaction": "fraud",
        "card fraud": "fraud",
        "transaction fraud": "fraud",
        "suspicious transaction": "fraud",

        "refund": "refund",
        "refund dispute": "refund",
        "dispute": "refund",
        "chargeback": "refund",

        "kyc": "kyc",
        "kyc verification": "kyc",

        "loan": "loan",
        "loan processing": "loan",
        "loan application": "loan",

        "general": "general",
        "other": "general",
    }

    if x in aliases:
        return aliases[x]

    if any(k in x for k in ["fraud", "unauthor", "suspicious"]):
        return "fraud"

    if any(k in x for k in ["refund", "dispute", "chargeback"]):
        return "refund"

    if "kyc" in x or "know your customer" in x:
        return "kyc"

    if "loan" in x:
        return "loan"

    return x if x else "general"


def risk_level(label: str) -> str:
    """Normalize risk prediction to HIGH / MEDIUM / LOW."""

    x = str(label).lower().strip()

    if "high" in x:
        return "HIGH"

    if "medium" in x or "moderate" in x:
        return "MEDIUM"

    if "low" in x:
        return "LOW"

    return str(label).upper() if label else "MEDIUM"


# ============================================================
# 7. DOCUMENT CATEGORY DETECTION
# ============================================================

def infer_document_category(source: str, text: str) -> str:
    """
    Your Data folder is flat, so category is inferred from
    filename + document content.
    """

    x = f"{source} {text[:2000]}".lower()

    if any(
        k in x
        for k in [
            "fraud",
            "unauthorized",
            "unauthorised",
            "suspicious transaction",
        ]
    ):
        return "fraud"

    if any(
        k in x
        for k in ["refund", "dispute", "chargeback"]
    ):
        return "refund"

    if "kyc" in x or "know your customer" in x:
        return "kyc"

    if "loan" in x:
        return "loan"

    return "general"


# ============================================================
# 8. TEXT / JSON HELPERS
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 450,
    overlap: int = 80,
) -> List[str]:

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def json_to_text(obj: Any) -> str:

    if isinstance(obj, dict):
        lines = []

        for key, value in obj.items():
            lines.append(
                f"{key}: {json_to_text(value)}"
            )

        return "\n".join(lines)

    if isinstance(obj, list):
        return "\n".join(
            json_to_text(item)
            for item in obj
        )

    return str(obj)


# ============================================================
# 9. LOAD ALL ML MODELS
# ============================================================

@st.cache_resource
def load_ml_models():

    return {
        "intent_model": load_pickle(
            INTENT_MODEL_PATH
        ),

        "intent_tfidf": load_pickle(
            INTENT_TFIDF_PATH
        ),

        "risk_model": load_pickle(
            RISK_MODEL_PATH
        ),

        "risk_tfidf": load_pickle(
            RISK_TFIDF_PATH
        ),

        "sentiment_model": load_pickle(
            SENTIMENT_MODEL_PATH
        ),

        "sentiment_tfidf": load_pickle(
            SENTIMENT_TFIDF_PATH
        ),
    }


models = load_ml_models()


# ============================================================
# 10. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


try:
    embedding_model = load_embedding_model()
    embedding_error = None

except Exception as exc:
    embedding_model = None
    embedding_error = str(exc)


# ============================================================
# 11. READ BANKING DOCUMENTS
# ============================================================

def read_data_files() -> List[Dict[str, str]]:

    documents = []

    if not DATA_DIR.exists():
        return documents

    for path in DATA_DIR.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:

            if suffix == ".txt":

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

            elif suffix == ".md":

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

            elif suffix == ".json":

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    text = json_to_text(
                        json.load(f)
                    )

            elif suffix == ".pdf":

                reader = PdfReader(str(path))

                pages = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]

                text = "\n".join(pages)

            else:
                continue

            if text.strip():

                documents.append(
                    {
                        "source": path.name,
                        "path": str(
                            path.relative_to(DATA_DIR)
                        ),
                        "text": text,
                        "category":
                            infer_document_category(
                                path.name,
                                text,
                            ),
                    }
                )

        except Exception as exc:

            print(
                f"Skipping {path.name}: {exc}"
            )

    return documents


# ============================================================
# 12. BUILD RAG KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def build_knowledge_base():

    if embedding_model is None:
        return None, []

    raw_docs = read_data_files()

    chunks = []

    for doc in raw_docs:

        for chunk in chunk_text(
            doc["text"]
        ):

            chunks.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "path": doc["path"],
                    "category": doc["category"],
                }
            )

    if not chunks:
        return None, []

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, chunks


vector_index, chunks = build_knowledge_base()


# ============================================================
# 13. RAG RETRIEVER
# ============================================================

def retrieve(
    query: str,
    category: str,
    top_k: int = 5,
):

    if (
        vector_index is None
        or not chunks
        or embedding_model is None
    ):
        return []

    q_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(
        max(top_k * 8, 20),
        len(chunks),
    )

    scores, indices = vector_index.search(
        q_embedding,
        search_k,
    )

    category_results = []
    fallback_results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        item = chunks[idx]

        result = {
            **item,
            "score": float(score),
        }

        fallback_results.append(result)

        if item["category"] == category:
            category_results.append(result)

        if len(category_results) >= top_k:
            break

    return (
        category_results[:top_k]
        or fallback_results[:top_k]
    )


# ============================================================
# 14. RECOMMENDED ACTION
# ============================================================

def recommended_action(
    category: str,
    risk: str,
) -> str:

    cat = str(category).lower()
    risk = str(risk).upper()

    if cat == "fraud" or "fraud" in cat:

        if risk == "HIGH":
            return (
                "Block/restrict the card or transaction "
                "and escalate to the fraud investigation team."
            )

        if risk == "MEDIUM":
            return (
                "Verify the transaction with the customer "
                "and consider temporary card restriction."
            )

        return (
            "Verify the transaction with the customer "
            "and follow the fraud-handling policy."
        )

    if cat == "refund":
        return (
            "Verify the transaction/dispute details and "
            "follow the refund or dispute policy."
        )

    if cat == "kyc":
        return (
            "Verify the customer's KYC information and "
            "follow the KYC policy."
        )

    if cat == "loan":
        return (
            "Verify the customer's loan details and "
            "follow the loan-processing policy."
        )

    return (
        "Review the relevant banking policy "
        "and provide customer support."
    )


# ============================================================
# 15. OLLAMA CLIENT
# ============================================================

@st.cache_resource
def get_ollama_client():

    return ollama.Client(
        host=OLLAMA_HOST
    )


def generate_answer(
    query: str,
    category: str,
    sentiment: str,
    risk: str,
    retrieved_docs: List[Dict[str, str]],
) -> str:

    context_parts = []

    for i, doc in enumerate(
        retrieved_docs,
        1,
    ):

        context_parts.append(
            f"[Document {i} | "
            f"{doc['source']} | "
            f"category={doc['category']}]\n"
            f"{doc['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    action = recommended_action(
        category,
        risk,
    )

    system_prompt = """
You are a banking customer-support assistant.

Use the retrieved banking documents as the source of truth.

Do not invent banking policies, fees, timelines,
transaction details, or guarantees.

If the documents are insufficient, explicitly say that
additional verification is required.

For possible fraud or unauthorized transactions,
prioritize customer safety and recommend escalation.

Return:
1. Customer Response
2. Recommended Next Action

Keep the response concise, professional and easy to understand.
"""

    user_prompt = f"""
Customer query:
{query}

Predicted intent/category:
{category}

Predicted sentiment:
{sentiment}

Predicted risk:
{risk}

Recommended action:
{action}

Retrieved banking knowledge:
{context}

Generate the final answer for the customer.
"""

    try:

        client = get_ollama_client()

        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.1,
            },
        )

        return response["message"]["content"]

    except Exception as exc:

        return (
            "The local LLM could not generate the response.\n\n"
            f"Error: {exc}\n\n"
            "Please check that Ollama is running and "
            f"that '{OLLAMA_MODEL}' is installed."
        )


# ============================================================
# 16. SIDEBAR SYSTEM STATUS
# ============================================================

with st.sidebar:

    st.header("⚙️ System Status")

    status_items = [
        (
            "Intent Model",
            models["intent_model"],
        ),
        (
            "Intent TF-IDF",
            models["intent_tfidf"],
        ),
        (
            "Risk Model",
            models["risk_model"],
        ),
        (
            "Risk TF-IDF",
            models["risk_tfidf"],
        ),
        (
            "Sentiment Model",
            models["sentiment_model"],
        ),
        (
            "Sentiment TF-IDF",
            models["sentiment_tfidf"],
        ),
        (
            "RAG Knowledge Base",
            vector_index,
        ),
    ]

    for name, obj in status_items:

        if obj is not None:
            st.success(
                f"✅ {name}"
            )
        else:
            st.warning(
                f"⚠️ {name} missing"
            )

    st.info(
        f"LLM: Ollama / {OLLAMA_MODEL}"
    )

    st.info(
        f"Embedding: {EMBEDDING_MODEL}"
    )

    st.write(
        f"Indexed chunks: {len(chunks)}"
    )

    if embedding_error:
        st.error(
            f"Embedding model error: {embedding_error}"
        )

    st.divider()

    st.caption(
        f"Project root:\n{BASE_DIR}"
    )

    st.caption(
        f"Model folder:\n{MODEL_DIR}"
    )


# ============================================================
# 17. CUSTOMER QUERY
# ============================================================

st.subheader("💬 Customer Query")

query = st.text_area(
    "Enter the customer's banking query:",
    placeholder=(
        "Example: I see a transaction of "
        "₹10,000 I didn't make"
    ),
    height=120,
)


# ============================================================
# 18. ANALYZE QUERY
# ============================================================

if st.button(
    "🔍 Analyze Query",
    type="primary",
    use_container_width=True,
):

    if not query.strip():

        st.warning(
            "Please enter a customer query."
        )

        st.stop()

    # --------------------------------------------------------
    # ML PREDICTIONS
    # --------------------------------------------------------

    with st.spinner(
        "Running ML models..."
    ):

        intent = predict_with_model(
            models["intent_model"],
            models["intent_tfidf"],
            query,
            "General",
        )

        sentiment = predict_with_model(
            models["sentiment_model"],
            models["sentiment_tfidf"],
            query,
            "Neutral",
        )

        risk_raw = predict_with_model(
            models["risk_model"],
            models["risk_tfidf"],
            query,
            "Medium",
        )

    category = canonical_category(
        intent
    )

    risk = risk_level(
        risk_raw
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving relevant banking documents..."
    ):

        retrieved_docs = retrieve(
            query=query,
            category=category,
            top_k=5,
        )

    action = recommended_action(
        category,
        risk,
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    with st.spinner(
        f"Generating response with {OLLAMA_MODEL}..."
    ):

        answer = generate_answer(
            query=query,
            category=category,
            sentiment=sentiment,
            risk=risk,
            retrieved_docs=retrieved_docs,
        )

    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader("📊 AI Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Intent",
            intent,
        )

    with col2:
        st.metric(
            "Sentiment",
            sentiment,
        )

    with col3:

        if risk == "HIGH":
            st.error(
                f"⚠️ RISK: {risk}"
            )

        else:
            st.metric(
                "Risk Level",
                risk,
            )

    st.write(
        f"**RAG Category Filter:** `{category}`"
    )

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    st.subheader(
        "🤖 AI Generated Response"
    )

    st.info(answer)

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    st.subheader(
        "🚨 Recommended Next Action"
    )

    if risk == "HIGH":

        st.error(action)

    elif risk == "MEDIUM":

        st.warning(action)

    else:

        st.success(action)

    # --------------------------------------------------------
    # RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    st.subheader(
        "📚 Retrieved Documents"
    )

    if retrieved_docs:

        for i, doc in enumerate(
            retrieved_docs,
            1,
        ):

            with st.expander(
                f"{i}. {doc['source']} | "
                f"{doc['category']} | "
                f"score={doc['score']:.3f}"
            ):

                st.write(
                    doc["text"]
                )

    else:

        st.warning(
            "No relevant documents were found."
        )
