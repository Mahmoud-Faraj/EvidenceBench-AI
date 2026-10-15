from __future__ import annotations

import re

from .retrieval import tokenize
from .schemas import Answer, RetrievedDocument

SENTENCE_PATTERN = re.compile(r"(?<=[.!?؟])\s+")


class ExtractiveGroundedGenerator:
    """Deterministic offline baseline that only returns retrieved evidence."""

    def __init__(self, minimum_confidence: float = 0.24, maximum_sentences: int = 3):
        self.minimum_confidence = minimum_confidence
        self.maximum_sentences = maximum_sentences

    def generate(self, question: str, contexts: list[RetrievedDocument]) -> Answer:
        if not contexts or contexts[0].score < self.minimum_confidence:
            return Answer(
                text="I do not have sufficient evidence in the retrieved collection to answer reliably.",
                citations=(),
                confidence=float(contexts[0].score if contexts else 0.0),
                abstained=True,
                reason="retrieval confidence below the configured threshold",
            )

        query_terms = set(tokenize(question))
        candidates: list[tuple[float, str, str]] = []
        for context in contexts:
            if context.score <= 0:
                continue
            sentences = SENTENCE_PATTERN.split(context.document.text)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                terms = set(tokenize(sentence))
                overlap = len(query_terms & terms) / max(len(query_terms), 1)
                score = 0.65 * context.score + 0.35 * overlap
                if overlap > 0:
                    candidates.append((score, sentence, context.document.doc_id))

        if not candidates:
            return Answer(
                text="I do not have sufficient evidence in the retrieved collection to answer reliably.",
                citations=(),
                confidence=float(contexts[0].score),
                abstained=True,
                reason="retrieved passages did not contain query-aligned evidence",
            )

        selected: list[tuple[float, str, str]] = []
        seen_sentences: set[str] = set()
        for item in sorted(candidates, reverse=True):
            if item[1] in seen_sentences:
                continue
            selected.append(item)
            seen_sentences.add(item[1])
            if len(selected) == self.maximum_sentences:
                break

        citations = tuple(dict.fromkeys(item[2] for item in selected))
        text = " ".join(
            f"{sentence.rstrip('.!?؟')} [{doc_id}]{sentence[-1] if sentence[-1] in '.!?؟' else '.'}"
            for _, sentence, doc_id in selected
        )
        confidence = float(sum(item[0] for item in selected) / len(selected))
        return Answer(text=text, citations=citations, confidence=confidence, abstained=False)
