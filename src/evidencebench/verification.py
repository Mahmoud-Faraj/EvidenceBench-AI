from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import tokenize
from .schemas import Answer, RetrievedDocument

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


@dataclass(frozen=True)
class VerificationResult:
    citation_precision: float
    supported_claim_ratio: float
    invalid_citations: tuple[str, ...]


def verify_answer(answer: Answer, contexts: list[RetrievedDocument]) -> VerificationResult:
    documents = {context.document.doc_id: context.document.text for context in contexts}
    cited_ids = CITATION_PATTERN.findall(answer.text)
    if not cited_ids:
        return VerificationResult(0.0, 0.0, ())

    invalid = tuple(sorted(set(cited_ids) - set(documents)))
    valid_count = sum(citation in documents for citation in cited_ids)
    citation_precision = valid_count / len(cited_ids)

    claims = [claim.strip() for claim in re.split(r"(?<=[.!?؟])\s+", answer.text) if claim.strip()]
    supported = 0
    assessed = 0
    for claim in claims:
        claim_citations = CITATION_PATTERN.findall(claim)
        claim_text = CITATION_PATTERN.sub("", claim)
        if not claim_citations:
            continue
        assessed += 1
        claim_terms = set(tokenize(claim_text))
        evidence_terms: set[str] = set()
        for citation in claim_citations:
            evidence_terms.update(tokenize(documents.get(citation, "")))
        overlap = len(claim_terms & evidence_terms) / max(len(claim_terms), 1)
        supported += overlap >= 0.6
    supported_ratio = supported / assessed if assessed else 0.0
    return VerificationResult(citation_precision, supported_ratio, invalid)
