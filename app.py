"""
AI Clinical Decision Support Lite — FastAPI Web Backend
------------------------------------------------------
Serves REST API endpoints for the clinical RAG pipeline and hosts the static frontend UI.

Endpoints:
  GET  /        -> Web UI Homepage
  GET  /health  -> Vector database & system health status
  POST /ask     -> End-to-end RAG query (Retrieval + Grounded Generation)
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from query import load_index, retrieve
from generate import generate_grounded_answer

app = FastAPI(
    title="AI Clinical Decision Support Lite",
    description="Grounded Clinical Evidence Q&A RAG System",
    version="1.0.0"
)

# Define static directory
STATIC_DIR = config.BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files (CSS, JS, assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker ('user' or 'assistant')")
    content: str = Field(..., description="Message text content")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Clinical question to ask")
    top_k: Optional[int] = Field(default=None, description="Number of evidence chunks to retrieve")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Recent conversation history")


class RetrievedChunkInfo(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    score: float
    snippet: str


class CitationInfo(BaseModel):
    document: str
    section: Optional[str] = "N/A"
    page: int


class RAGResponse(BaseModel):
    question: str
    recommendation: str
    evidence: str
    citations: List[CitationInfo]
    confidence: str
    retrieved_chunks: List[RetrievedChunkInfo]


@app.get("/", response_class=FileResponse)
def read_root():
    """Serves the main frontend index.html page."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))


@app.get("/health")
def health_check():
    """Checks system readiness and ChromaDB index existence."""
    db_exists = config.CHROMA_DIR.exists()
    has_api_key = bool(config.GEMINI_API_KEY)
    
    return {
        "status": "ready" if db_exists else "database_missing",
        "chroma_db_exists": db_exists,
        "chroma_dir": str(config.CHROMA_DIR),
        "gemini_api_key_configured": has_api_key,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "gemini_model": config.GEMINI_MODEL
    }


@app.post("/ask", response_model=RAGResponse)
def ask_question(req: QuestionRequest):
    """Executes retrieval + grounded generation for a clinical question."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert Pydantic ChatMessage objects to dicts
    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in (req.chat_history or [])
    ]

    # Step 1: Context-aware Retrieval
    try:
        vectordb = load_index()
        results = retrieve(vectordb, question, k=req.top_k, chat_history=history_dicts)
    except SystemExit:
        raise HTTPException(
            status_code=500,
            detail="Vector database is missing. Please run 'python ingest.py' first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {str(e)}"
        )

    # Format retrieved chunks for response UI
    retrieved_chunks = []
    for doc, score in results:
        meta = doc.metadata
        retrieved_chunks.append(
            RetrievedChunkInfo(
                chunk_id=str(meta.get("chunk_id", "N/A")),
                document_name=str(meta.get("document_name", "Unknown")),
                page_number=int(meta.get("page_number", 1)),
                score=round(float(score), 4),
                snippet=doc.page_content.strip()
            )
        )

    # Step 2: Grounded Generation with Memory
    try:
        gen_response = generate_grounded_answer(question, results, chat_history=history_dicts)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


    citations = [
        CitationInfo(
            document=c.get("document", "Unknown"),
            section=c.get("section", "N/A"),
            page=int(c.get("page", 1))
        )
        for c in gen_response.get("citations", [])
    ]

    return RAGResponse(
        question=question,
        recommendation=gen_response.get("recommendation", ""),
        evidence=gen_response.get("evidence", ""),
        citations=citations,
        confidence=gen_response.get("confidence", "insufficient"),
        retrieved_chunks=retrieved_chunks
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
