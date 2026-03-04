"""
OSB RAG Pipeline
Retrieval-Augmented Generation:
1. Embed the user query
2. Retrieve top-K relevant chunks from FAISS
3. Build a context-aware prompt
4. Call Gemini (or Claude fallback) for the answer
5. Return answer + sources
"""

import os
import time
import json
from typing import List, Dict
from datetime import datetime

import numpy as np
import google.generativeai as genai

from embeddings import get_embedder, search_index, get_index_stats
from ingest import load_chunks, load_documents

# ── Config ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QUERY_LOG_FILE = "osb_data/query_log.json"

# Stats tracker
_query_count = 0
_total_time = 0.0


# ── Gemini setup ──────────────────────────────────────────────
def get_gemini():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://makersuite.google.com/app/apikey"
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


# ── Main RAG function ─────────────────────────────────────────
def query_knowledge_base(question: str, top_k: int = 5) -> Dict:
    """
    Full RAG pipeline:
    1. Embed question
    2. Retrieve relevant chunks
    3. Generate answer with Gemini
    4. Return structured response
    """
    global _query_count, _total_time
    start = time.time()
    _query_count += 1

    # Step 1: Embed the question
    embedder = get_embedder()
    q_embedding = embedder.encode([question], show_progress_bar=False)[0]
    q_embedding = np.array(q_embedding, dtype=np.float32)

    # Step 2: Retrieve top-K chunks
    results = search_index(q_embedding, top_k=top_k)

    if not results:
        return {
            "answer": "No documents indexed yet. Please upload some files first using the /upload endpoint.",
            "sources": [],
            "retrieved_chunks": 0
        }

    # Step 3: Load chunk texts
    all_chunks = load_chunks()
    chunk_map = {c["id"]: c for c in all_chunks}

    retrieved = []
    for chunk_id, score in results:
        if chunk_id in chunk_map:
            chunk = chunk_map[chunk_id]
            retrieved.append({
                "text": chunk["text"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "score": round(score, 4)
            })

    if not retrieved:
        return {
            "answer": "Could not retrieve relevant context. Try re-indexing your documents.",
            "sources": [],
            "retrieved_chunks": 0
        }

    # Step 4: Build prompt
    context = "\n\n".join([
        f"[Source: {r['filename']} | Chunk {r['chunk_index']+1}/{r['total_chunks']} | Relevance: {r['score']:.2f}]\n{r['text']}"
        for r in retrieved
    ])

    prompt = f"""You are OSB — Operating System Brain, a precise AI knowledge agent.
You answer questions STRICTLY based on the provided document context.

CONTEXT FROM INDEXED DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer precisely based only on the context above
- If the context doesn't contain enough information, say so clearly
- Quote relevant passages when helpful
- Be concise and technical
- Do NOT make up information not present in the context

ANSWER:"""

    # Step 5: Call Gemini
    try:
        model = get_gemini()
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"AI model error: {str(e)}\n\nRetrieved context:\n\n" + "\n---\n".join([r["text"] for r in retrieved[:2]])

    # Step 6: Build sources list
    seen = set()
    sources = []
    for r in retrieved:
        key = f"{r['filename']}_{r['chunk_index']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": r["filename"],
                "chunk": f"Chunk {r['chunk_index']+1} of {r['total_chunks']}",
                "relevance": r["score"],
                "preview": r["text"][:120] + "..."
            })

    elapsed = round(time.time() - start, 2)
    _total_time += elapsed

    # Log query
    log_query(question, answer, sources, elapsed)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(retrieved),
        "query_time": elapsed
    }


# ── System stats ──────────────────────────────────────────────
def get_system_stats() -> Dict:
    docs = load_documents()
    chunks = load_chunks()
    idx_stats = get_index_stats()

    total_size = sum(d.get("size_bytes", 0) for d in docs.values())

    return {
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "vector_dimensions": idx_stats["dimensions"],
        "total_queries": _query_count,
        "avg_response_time": round(_total_time / max(_query_count, 1), 2),
        "storage_mb": round(total_size / (1024 * 1024), 2),
        "status": "operational"
    }


# ── Query logging ─────────────────────────────────────────────
def log_query(question: str, answer: str, sources: List, elapsed: float):
    try:
        log = []
        if os.path.exists(QUERY_LOG_FILE):
            with open(QUERY_LOG_FILE, "r") as f:
                log = json.load(f)
        log.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "sources_used": [s["filename"] for s in sources],
            "response_time": elapsed
        })
        # Keep last 100 queries
        log = log[-100:]
        with open(QUERY_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass
