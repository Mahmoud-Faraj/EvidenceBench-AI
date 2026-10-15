from evidencebench.pipeline import EvidencePipeline
from evidencebench.schemas import Document


def test_pipeline_returns_grounded_citations():
    pipeline = EvidencePipeline(
        [
            Document(
                "D1",
                "Hybrid retrieval",
                "Hybrid retrieval combines lexical matching with semantic search.",
            ),
            Document("D2", "Unrelated", "Weather varies during the year."),
        ]
    )
    result = pipeline.answer("What does hybrid retrieval combine?", top_k=2)
    assert not result.answer.abstained
    assert "D1" in result.answer.citations
    assert result.verification.citation_precision == 1.0
    assert result.verification.supported_claim_ratio == 1.0


def test_pipeline_abstains_when_evidence_is_weak():
    pipeline = EvidencePipeline([Document("D1", "Apples", "Apples are fruit.")])
    result = pipeline.answer("Explain quantum entanglement")
    assert result.answer.abstained
