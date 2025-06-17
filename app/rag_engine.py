from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from ctransformers import AutoModelForCausalLM
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Use a lightweight local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

nltk.download("punkt_tab")
nltk.download("stopwords")

# Load and split documents
def prepare_documents(filepath='pdf'):
    loader = PyPDFDirectoryLoader(filepath)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    return splitter.split_documents(docs)

# Build or load FAISS vector DB
def create_vector_store(docs):
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local("vector_store")
    return db

def load_vector_store():
    return FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)

# Load LLM model once
llm_model = AutoModelForCausalLM.from_pretrained(
    "./models",                 # Local path inside Docker
    model_file="TinyLlama-1.1B-Chat.Q4_K_M.gguf",
    model_type="llama"
)

def clean_context(text, max_tokens):

    stop_words = set(stopwords.words("english"))
    # Normalize
    text = text.lower()
    
    # Remove special symbols
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
    
    # Remove stopwords
    words = word_tokenize(text)
    filtered = [w for w in words if w not in stop_words]
    
    # Reconstruct and truncate
    clean_text = " ".join(filtered)
    return clean_text[:max_tokens]  # or tokenize and limit to N tokens

# Perform retrieval-augmented search
def query_rag(vector_db, query: str, k: int = 5):
    relevant_docs = vector_db.similarity_search(query, k=k)
    sources = []
    answer_parts = []
    for doc in relevant_docs:
        meta = doc.metadata if doc.metadata else {}
        sources.append({
            "document": meta.get("source", "unknown"),
            "page": meta.get("page", 1),
            "relevance_score": 0.89  # Simulated
        })
        answer_parts.append(doc.page_content)

    combined_context = "\n---\n".join(answer_parts[:3])  # Use top 3 chunks as context
    cleaned_context = clean_context(combined_context,400)

    return cleaned_context, sources

# Perform RAG + local LLM generation
def generate_answer_with_ollama(query: str, context: str):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return llm_model(prompt)

# Health check
def check_model_health():
    try:
        llm_model("Hello!")  # Dummy call
        return "Healthy"
    except:
        return "Unhealthy"