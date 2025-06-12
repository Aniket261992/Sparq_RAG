from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import time
import psutil
import os

from app.rag_engine import prepare_documents, create_vector_store, query_rag, generate_answer_with_ollama

app = FastAPI()

# Initialize RAG pipeline
start_time = datetime.now()
documents = prepare_documents()
vector_db = create_vector_store(documents)

# Metrics
total_queries = 0
response_times = []

# --- Models ---
class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    include_sources: bool = True

class Source(BaseModel):
    document: str
    page: int
    relevance_score: float

class QueryResponse(BaseModel):
    status: str
    answer: str
    sources: List[Source]
    response_time_ms: int
    llm_generation_time_ms: int
    vector_search_time_ms: int
    used_llm: bool

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/stats")
def stats():
    uptime = str(datetime.now() - start_time).split('.')[0]
    avg_resp = int(sum(response_times) / len(response_times)) if response_times else 0
    mem = int(psutil.virtual_memory().used / (1024 ** 2))
    cpu = psutil.cpu_percent(interval=0.1)
    load = os.getloadavg() if hasattr(os, "getloadavg") else [0.0, 0.0, 0.0]
    return {
        "total_queries": total_queries,
        "avg_response_time_ms": avg_resp,
        "uptime": uptime,
        "memory_usage_mb": mem,
        "cpu_percent": cpu,
        "system_load": list(load)
    }

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    global total_queries
    start = time.time()
    context, sources = query_rag(vector_db, req.question, req.max_results)
    llm_start = time.time()
    answer = generate_answer_with_ollama(req.question, context)
    llm_time = int((time.time() - llm_start) * 1000)
    resp_time = int((time.time() - start) * 1000)

    response_times.append(resp_time)
    total_queries += 1

    return {
        "status": "success",
        "answer": answer,
        "sources": sources if req.include_sources else [],
        "response_time_ms": resp_time,
        "llm_generation_time_ms": llm_time,
        "vector_search_time_ms": resp_time - llm_time,
        "used_llm": True
    }

@app.get("/info")
def info():
    return {
        "team_name": "Project team name",
        "model_info": "mistral:latest with Ollama",
        "documents_processed": 42,
        "total_chunks": 1250,
        "vector_db": "ChromaDB",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 2000,
        "chunk_overlap": 400
    }