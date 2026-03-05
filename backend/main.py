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
            fname = f.filename.lower()
            text = ""
            try:
                if fname.endswith(".pdf"):
                    import fitz
                    doc = fitz.open(stream=raw, filetype="pdf")
                    text = "\n".join(page.get_text() for page in doc)
                    print(f"PDF extracted: {len(text)} chars")
                elif fname.endswith((".docx", ".doc")):
                    import docx, io
                    doc = docx.Document(io.BytesIO(raw))
                    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    print(f"DOCX extracted: {len(text)} chars")
                elif fname.endswith((".xlsx", ".xls")):
                    # Basic Excel - extract as CSV-like text
                    text = f"[Excel file: {f.filename} - contains spreadsheet data]"
                    try:
                        import zipfile, io as sio
                        z = zipfile.ZipFile(sio.BytesIO(raw))
                        for name in z.namelist():
                            if name.endswith('.xml') and 'sheet' in name:
                                xml = z.read(name).decode('utf-8', errors='ignore')
                                import re
                                vals = re.findall(r'<v>([^<]+)</v>', xml)
                                text = f"[Excel data from {f.filename}]: " + ", ".join(vals[:200])
                    except: pass
                elif fname.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    text = f"[Image file: {f.filename} - visual content, cannot extract text]"
                elif fname.endswith(".csv"):
                    text = raw.decode("utf-8", errors="ignore")[:6000]
                else:
                    # Try to decode as text for all other types
                    text = raw.decode("utf-8", errors="ignore")
                    print(f"Text read: {len(text)} chars")
            except Exception as ex:
                text = raw.decode("utf-8", errors="ignore")
                print(f"Fallback for {f.filename}: {ex}")
            if not text.strip():
                text = f"[File {f.filename} could not be read or is empty]"
            file_context += f"\n--- FILE: {f.filename} ---\n{text[:6000]}\n--- END ---\n"

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
