"""
OSB Embeddings & FAISS Vector Store
Manages sentence embeddings and similarity search.
"""

import os
import numpy as np
import pickle
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, 384-dim, great for RAG
DIMENSION = 384
DATA_DIR = "osb_data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# Singletons
_embedder = None
_index = None
_id_map = None      # maps faiss internal id → chunk id string
_next_id = 0


# ── Embedder ──────────────────────────────────────────────────
def get_embedder() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    global _embedder
    if _embedder is None:
        print(f"🔄 Loading embedding model: {MODEL_NAME}")
        _embedder = SentenceTransformer(MODEL_NAME)
        print("✅ Embedding model loaded")
    return _embedder


# ── FAISS Index ───────────────────────────────────────────────
def get_index():
    """Load or create FAISS index."""
    global _index, _id_map, _next_id
    if _index is None:
        if os.path.exists(INDEX_FILE) and os.path.exists(ID_MAP_FILE):
            _index = faiss.read_index(INDEX_FILE)
            with open(ID_MAP_FILE, "rb") as f:
                data = pickle.load(f)
                _id_map = data["id_map"]
                _next_id = data["next_id"]
            print(f"✅ Loaded FAISS index: {_index.ntotal} vectors")
        else:
            # Inner product index (works as cosine sim on normalised vectors)
            _index = faiss.IndexFlatIP(DIMENSION)
            _id_map = {}
            _next_id = 0
            print("✅ Created new FAISS index")
    return _index


def save_index():
    """Persist FAISS index and ID map to disk."""
    global _index, _id_map, _next_id
    if _index is not None:
        faiss.write_index(_index, INDEX_FILE)
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump({"id_map": _id_map, "next_id": _next_id}, f)


def add_to_index(embeddings: np.ndarray) -> List[int]:
    """Add embeddings to FAISS, return list of assigned IDs."""
    global _next_id, _id_map
    index = get_index()

    ids = list(range(_next_id, _next_id + len(embeddings)))
    for i in ids:
        _id_map[i] = i

    index.add(embeddings)
    _next_id += len(embeddings)
    save_index()
    return ids


def search_index(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Search FAISS index for most similar chunks.
    Returns list of (chunk_id, similarity_score) tuples.
    """
    index = get_index()
    if index.ntotal == 0:
        return []

    # Normalise query
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and score > 0.1:   # filter low relevance
            results.append((int(idx), float(score)))

    return results


def get_index_stats() -> dict:
    """Return current index statistics."""
    index = get_index()
    return {
        "total_vectors": index.ntotal,
        "dimensions": DIMENSION,
        "model": MODEL_NAME,
        "index_type": "IndexFlatIP (cosine similarity)"
    }
