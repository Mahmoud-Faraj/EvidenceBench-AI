"""Evaluate offline retrievers on the prepared SciFact development split."""

from __future__ import annotations

import json
from pathlib import Path

from evidencebench.data import load_documents, load_evaluation_set
from evidencebench.evaluation import evaluate_retriever
from evidencebench.retrieval import BM25Retriever, HybridRetriever, TfidfSemanticRetriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "scifact"


class RankedAdapter:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 10):
        scores = self.retriever.scores(query)
        order = scores.argsort()[::-1][:top_k]
        from evidencebench.schemas import RetrievedDocument

        return [
            RetrievedDocument(self.retriever.documents[index], float(scores[index]), rank)
            for rank, index in enumerate(order, start=1)
        ]


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python scripts/download_scifact.py` first.")
    documents = load_documents(DATA / "documents.json")
    examples = load_evaluation_set(DATA / "evaluation.json")
    retrievers = {
        "bm25": RankedAdapter(BM25Retriever(documents)),
        "tfidf": RankedAdapter(TfidfSemanticRetriever(documents)),
        "hybrid": HybridRetriever(documents),
    }
    results = {
        name: evaluate_retriever(retriever, examples, top_k=10) for name, retriever in retrievers.items()
    }
    output = ROOT / "outputs" / "scifact_retrieval_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
