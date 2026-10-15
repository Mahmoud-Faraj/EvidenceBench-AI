from __future__ import annotations

import math
from statistics import mean

from .schemas import EvaluationExample, RetrievedDocument


def reciprocal_rank(results: list[RetrievedDocument], relevant_ids: set[str]) -> float:
    for result in results:
        if result.document.doc_id in relevant_ids:
            return 1.0 / result.rank
    return 0.0


def recall_at_k(results: list[RetrievedDocument], relevant_ids: set[str], k: int) -> float:
    retrieved = {result.document.doc_id for result in results[:k]}
    return len(retrieved & relevant_ids) / max(len(relevant_ids), 1)


def ndcg_at_k(results: list[RetrievedDocument], relevant_ids: set[str], k: int) -> float:
    gains = [1.0 if result.document.doc_id in relevant_ids else 0.0 for result in results[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever(retriever, examples: list[EvaluationExample], top_k: int = 5) -> dict[str, float]:
    rows = []
    for example in examples:
        results = retriever.retrieve(example.question, top_k=top_k)
        relevant = set(example.relevant_doc_ids)
        rows.append(
            {
                "mrr": reciprocal_rank(results, relevant),
                "recall_at_k": recall_at_k(results, relevant, top_k),
                "ndcg_at_k": ndcg_at_k(results, relevant, top_k),
            }
        )
    return {key: mean(row[key] for row in rows) for key in rows[0]}
