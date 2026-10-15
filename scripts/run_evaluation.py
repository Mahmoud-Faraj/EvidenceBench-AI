from __future__ import annotations

import json
from pathlib import Path

from evidencebench.data import load_documents, load_evaluation_set
from evidencebench.evaluation import evaluate_retriever
from evidencebench.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_documents(ROOT / "data" / "documents.json")
    examples = load_evaluation_set(ROOT / "data" / "evaluation.json")
    output = {}
    for weight in (0.0, 0.5, 1.0):
        name = {0.0: "tfidf", 0.5: "hybrid", 1.0: "bm25"}[weight]
        output[name] = evaluate_retriever(
            HybridRetriever(documents, lexical_weight=weight), examples, top_k=5
        )
    path = ROOT / "outputs" / "retrieval_metrics.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
