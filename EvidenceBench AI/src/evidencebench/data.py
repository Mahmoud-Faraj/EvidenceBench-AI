from __future__ import annotations

import json
from pathlib import Path

from .schemas import Document, EvaluationExample


def load_documents(path: str | Path) -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = [Document(**record) for record in records]
    if not documents:
        raise ValueError("The document collection is empty")
    ids = [document.doc_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique")
    return documents


def load_evaluation_set(path: str | Path) -> list[EvaluationExample]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvaluationExample(
            question_id=record["question_id"],
            question=record["question"],
            relevant_doc_ids=tuple(record["relevant_doc_ids"]),
            language=record.get("language", "en"),
        )
        for record in records
    ]
