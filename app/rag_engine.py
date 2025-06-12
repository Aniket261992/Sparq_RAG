from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import ollama

# Use a lightweight local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load and split documents
def prepare_documents(filepath='app/speech.txt'):
    loader = TextLoader(filepath)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_documents(docs)

# Build or load FAISS vector DB
def create_vector_store(docs):
    return FAISS.from_documents(docs, embedding_model)

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
    return combined_context, sources

# Call local LLM using Ollama
def generate_answer_with_ollama(query: str, context: str, model: str = "mistral"):
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
