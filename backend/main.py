"""
OSB — Operating System Brain
FastAPI Backend Server
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="OSB API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "OSB API is running", "status": "operational"}

@app.get("/health")
def health():
    key = os.getenv("GROQ_API_KEY", "")
    return {"status": "ok", "groq_key_set": bool(key), "key_preview": key[:8] + "..." if key else "NOT SET"}

@app.post("/query-with-files")
async def query_with_files(question: str = Form(...), files: List[UploadFile] = File(default=[])):
    print(f"Received question: {question}")
    print(f"Received files: {[f.filename for f in files]}")
    
    try:
        from groq import Groq
        
        groq_key = os.getenv("GROQ_API_KEY", "")
        print(f"GROQ key present: {bool(groq_key)}")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

        file_context = ""
        for f in files:
            raw = await f.read()
            text = raw.decode("utf-8", errors="ignore")
            file_context += f"\n--- FILE: {f.filename} ---\n{text[:6000]}\n--- END ---\n"
            print(f"Read file {f.filename}: {len(text)} chars")

        system_prompt = f"""You are OSB — Operating System Brain.

PART 1 — FROM THE FILE:
Extract and quote relevant info from the file below.

PART 2 — ADDITIONAL CONTEXT:
Expand with your broader knowledge on the same topic.

FILE CONTENTS:
{file_context}"""

        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        sources = [f.filename for f in files]
        print(f"Answer generated: {len(answer)} chars")
        return {"answer": answer, "sources": sources}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("EXCEPTION:", tb)
        raise HTTPException(status_code=500, detail=type(e).__name__ + ": " + str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
