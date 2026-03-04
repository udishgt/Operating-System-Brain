"""
OSB Ingest Pipeline
Handles file parsing, chunking, embedding, and FAISS indexing.
Supports: PDF, DOCX, TXT, MD, Python, JS, JSON, CSV and more.
"""

import os
import io
import json
import uuid
import hashlib
import pickle
import numpy as np
from typing import List, Dict
from datetime import datetime

# ── Optional imports (graceful fallback) ─────────────────────
try:
    import fitz  # PyMuPDF for PDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

from embeddings import get_embedder, add_to_index

# ── Storage paths ─────────────────────────────────────────────
DATA_DIR = "osb_data"
DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Document store ────────────────────────────────────────────
def load_documents() -> Dict:
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_documents(docs: Dict):
    with open(DOCS_FILE, "w") as f:
        json.dump(docs, f, indent=2)


def load_chunks() -> List[Dict]:
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            return pickle.load(f)
    return []


def save_chunks(chunks: List[Dict]):
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)


# ── Text extraction ───────────────────────────────────────────
def extract_text(filename: str, content: bytes, ext: str) -> str:
    """Extract raw text from various file formats."""

    if ext in [".txt", ".md", ".py", ".js", ".ts", ".html",
               ".css", ".yaml", ".yml", ".rs", ".go", ".sh"]:
        return content.decode("utf-8", errors="ignore")

    elif ext == ".json":
        try:
            data = json.loads(content.decode("utf-8"))
            return json.dumps(data, indent=2)
        except Exception:
            return content.decode("utf-8", errors="ignore")

    elif ext == ".csv":
        return content.decode("utf-8", errors="ignore")

    elif ext == ".pdf":
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support requires PyMuPDF: pip install pymupdf")
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            text += f"\n[Page {page_num + 1}]\n"
            text += page.get_text()
        return text

    elif ext == ".docx":
        if not DOCX_SUPPORT:
            raise RuntimeError("DOCX support requires python-docx: pip install python-docx")
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    else:
        return content.decode("utf-8", errors="ignore")


# ── Chunking ──────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better context retrieval.
    chunk_size: characters per chunk
    overlap: characters shared between adjacent chunks
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 20]  # filter tiny chunks


# ── Main ingest function ──────────────────────────────────────
def ingest_file(filename: str, content: bytes, ext: str) -> Dict:
    """
    Full ingestion pipeline:
    1. Extract text
    2. Chunk text
    3. Embed chunks
    4. Add to FAISS index
    5. Save metadata
    """
    doc_id = str(uuid.uuid4())[:8]
    file_hash = hashlib.md5(content).hexdigest()

    # Check for duplicate
    docs = load_documents()
    for existing_id, doc in docs.items():
        if doc.get("hash") == file_hash:
            return {"doc_id": existing_id, "chunks": doc["chunks"], "duplicate": True}

    # Extract text
    raw_text = extract_text(filename, content, ext)
    if not raw_text.strip():
        raise ValueError(f"Could not extract text from {filename}")

    # Chunk
    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError(f"No chunks generated from {filename}")

    # Embed and index
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    # Add to FAISS
    chunk_ids = add_to_index(embeddings)

    # Save chunk metadata
    all_chunks = load_chunks()
    for i, (chunk_text_val, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        all_chunks.append({
            "id": chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "text": chunk_text_val,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    save_chunks(all_chunks)

    # Save document metadata
    docs[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "ext": ext,
        "size_bytes": len(content),
        "chunks": len(chunks),
        "hash": file_hash,
        "indexed_at": datetime.now().isoformat(),
        "chunk_ids": chunk_ids
    }
    save_documents(docs)

    print(f"✅ Indexed: {filename} → {len(chunks)} chunks (doc_id: {doc_id})")
    return {"doc_id": doc_id, "chunks": len(chunks)}


# ── Document management ───────────────────────────────────────
def get_document_list() -> List[Dict]:
    docs = load_documents()
    result = []
    for doc_id, doc in docs.items():
        result.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "ext": doc["ext"],
            "size_kb": round(doc["size_bytes"] / 1024, 1),
            "chunks": doc["chunks"],
            "indexed_at": doc["indexed_at"]
        })
    return sorted(result, key=lambda x: x["indexed_at"], reverse=True)


def delete_document(doc_id: str) -> bool:
    docs = load_documents()
    if doc_id not in docs:
        return False

    # Remove chunks
    chunk_ids_to_remove = set(docs[doc_id]["chunk_ids"])
    all_chunks = load_chunks()
    all_chunks = [c for c in all_chunks if c["id"] not in chunk_ids_to_remove]
    save_chunks(all_chunks)

    # Remove doc
    del docs[doc_id]
    save_documents(docs)
    return True
