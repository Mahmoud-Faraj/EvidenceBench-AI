from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    language: str = "en"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    rank: int
    lexical_score: float = 0.0
    semantic_score: float = 0.0


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    confidence: float
    abstained: bool
    reason: str = ""


@dataclass(frozen=True)
class EvaluationExample:
    question_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    language: str = "en"
