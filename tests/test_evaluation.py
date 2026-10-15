from evidencebench.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from evidencebench.schemas import Document, RetrievedDocument


def ranked_results():
    return [
        RetrievedDocument(Document("D1", "One", "Text"), 1.0, 1),
        RetrievedDocument(Document("D2", "Two", "Text"), 0.8, 2),
    ]


def test_ranking_metrics():
    results = ranked_results()
    assert reciprocal_rank(results, {"D2"}) == 0.5
    assert recall_at_k(results, {"D2"}, 1) == 0.0
    assert recall_at_k(results, {"D2"}, 2) == 1.0
    assert 0 < ndcg_at_k(results, {"D2"}, 2) < 1
