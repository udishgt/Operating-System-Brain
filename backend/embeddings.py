"""
OSB Embeddings - Stub for Railway deployment
FAISS and sentence-transformers run locally only.
On Railway, the /query endpoint uses Groq directly without vector search.
"""

import os

def get_embedder():
    return None

def add_to_index(chunks, doc_id):
    return 0

def search_index(query, top_k=5):
    return []

def get_index_stats():
    return {"total_vectors": 0, "dimensions": 384, "index_type": "FAISS (local only)"}
