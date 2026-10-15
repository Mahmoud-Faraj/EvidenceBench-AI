from __future__ import annotations

from dataclasses import dataclass

from .generation import ExtractiveGroundedGenerator
from .retrieval import HybridRetriever
from .schemas import Answer, Document, RetrievedDocument
from .verification import VerificationResult, verify_answer


@dataclass(frozen=True)
class PipelineResult:
    answer: Answer
    contexts: list[RetrievedDocument]
    verification: VerificationResult


class EvidencePipeline:
    def __init__(self, documents: list[Document], lexical_weight: float = 0.5):
        self.retriever = HybridRetriever(documents, lexical_weight=lexical_weight)
        self.generator = ExtractiveGroundedGenerator()

    def answer(self, question: str, top_k: int = 5) -> PipelineResult:
        contexts = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, contexts)
        verification = verify_answer(answer, contexts)
        return PipelineResult(answer, contexts, verification)
