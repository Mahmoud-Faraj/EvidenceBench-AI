from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evidencebench.data import load_documents  # noqa: E402
from evidencebench.pipeline import EvidencePipeline  # noqa: E402

app = FastAPI(title="EvidenceBench AI API", version="0.1.0")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


@lru_cache
def pipeline() -> EvidencePipeline:
    return EvidencePipeline(load_documents(ROOT / "data" / "documents.json"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer")
def answer(request: QuestionRequest) -> dict:
    result = pipeline().answer(request.question, top_k=request.top_k)
    return {
        "answer": result.answer.text,
        "abstained": result.answer.abstained,
        "confidence": result.answer.confidence,
        "citations": result.answer.citations,
        "citation_precision": result.verification.citation_precision,
        "supported_claim_ratio": result.verification.supported_claim_ratio,
        "contexts": [
            {
                "doc_id": context.document.doc_id,
                "title": context.document.title,
                "score": context.score,
            }
            for context in result.contexts
        ],
    }
