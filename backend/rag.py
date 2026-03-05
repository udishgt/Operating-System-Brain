"""
OSB RAG Pipeline - Railway version using Groq directly
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
from typing import Dict

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_query_count = 0
_total_time = 0.0

def query_knowledge_base(question: str, top_k: int = 5) -> Dict:
    global _query_count, _total_time
    from groq import Groq
    start = time.time()
    _query_count += 1

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = "Error: " + str(e)

    elapsed = time.time() - start
    _total_time += elapsed

    return {
        "answer": answer,
        "sources": [],
        "retrieved_chunks": 0,
        "query_time": round(elapsed, 2)
    }

def get_system_stats() -> Dict:
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "vector_dimensions": 384,
        "total_queries": _query_count,
        "avg_response_time": round(_total_time / max(_query_count, 1), 2),
        "storage_mb": 0.0,
        "status": "operational"
    }
