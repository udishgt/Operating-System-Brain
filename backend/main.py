"""
OSB — Operating System Brain
FastAPI Backend Server
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
import os

from ingest import ingest_file, get_document_list, delete_document
from rag import query_knowledge_base, get_system_stats

app = FastAPI(
    title="OSB — Operating System Brain API",
    description="Local-first RAG knowledge system API",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_time: float


class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    vector_dimensions: int
    total_queries: int
    avg_response_time: float
    storage_mb: float
    status: str


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "OSB API is running", "status": "operational"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document into the vector store."""
    allowed_types = [
        "text/plain", "text/markdown", "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/x-python", "application/json", "text/html", "text/css",
        "application/javascript", "text/csv"
    ]
    allowed_extensions = [
        ".txt", ".md", ".pdf", ".docx", ".py", ".js", ".ts",
        ".json", ".html", ".css", ".csv", ".yaml", ".yml", ".rs", ".go"
    ]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed_extensions)}"
        )

    try:
        content = await file.read()
        result = ingest_file(file.filename, content, ext)
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": result["chunks"],
            "doc_id": result["doc_id"],
            "message": f"Successfully indexed {result['chunks']} chunks from {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query the knowledge base using RAG."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    try:
        result = query_knowledge_base(req.question, top_k=req.top_k)
        elapsed = round(time.time() - start, 2)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_time=elapsed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    return {"documents": get_document_list()}


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    """Remove a document from the index."""
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "message": f"Document {doc_id} removed"}


@app.get("/stats", response_model=SystemStats)
def stats():
    """Get system stats for the dashboard."""
    return get_system_stats()


if __name__ == "__main__":
    print("\n🧠 OSB — Operating System Brain")
    print("=" * 40)
    print("Starting API server on http://localhost:8000")
    print("API docs at http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)