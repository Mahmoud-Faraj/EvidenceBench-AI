from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import Document, RetrievedDocument

TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(f"{document.title} {document.text}") for document in documents]
        self.lengths = np.asarray([len(tokens) for tokens in self.tokens], dtype=float)
        self.average_length = float(self.lengths.mean())
        self.frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.documents), dtype=float)
        for index, frequency in enumerate(self.frequencies):
            length_normalizer = 1 - self.b + self.b * self.lengths[index] / self.average_length
            for term in query_terms:
                term_frequency = frequency.get(term, 0)
                if not term_frequency:
                    continue
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * length_normalizer
                scores[index] += self.idf.get(term, 0.0) * numerator / denominator
        return scores


class TfidfSemanticRetriever:
    """Offline semantic-search baseline with an embedding-compatible interface."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            f"{document.title} {document.text}" for document in documents
        )

    def scores(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        return (self.matrix @ query_vector.T).toarray().ravel()


class HybridRetriever:
    def __init__(self, documents: list[Document], lexical_weight: float = 0.5):
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight must be between zero and one")
        self.documents = documents
        self.lexical_weight = lexical_weight
        self.bm25 = BM25Retriever(documents)
        self.semantic = TfidfSemanticRetriever(documents)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        minimum = scores.min()
        maximum = scores.max()
        if maximum <= minimum:
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        lexical = self._normalize(self.bm25.scores(query))
        semantic = self._normalize(self.semantic.scores(query))
        combined = self.lexical_weight * lexical + (1 - self.lexical_weight) * semantic
        order = np.argsort(-combined)[: min(top_k, len(self.documents))]
        return [
            RetrievedDocument(
                document=self.documents[index],
                score=float(combined[index]),
                rank=rank,
                lexical_score=float(lexical[index]),
                semantic_score=float(semantic[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]
