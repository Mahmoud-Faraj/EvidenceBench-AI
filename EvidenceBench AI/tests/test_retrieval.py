from evidencebench.retrieval import HybridRetriever, tokenize
from evidencebench.schemas import Document


def documents():
    return [
        Document("D1", "RAG", "Retrieval augmented generation uses external evidence."),
        Document("D2", "Metrics", "Recall and reciprocal rank evaluate retrieval."),
        Document("D3", "الامتناع", "يجب الامتناع عن الإجابة عندما لا توجد أدلة كافية.", "ar"),
    ]


def test_tokenizer_supports_english_and_arabic():
    assert "retrieval" in tokenize("Retrieval works")
    assert "الأدلة" in tokenize("لا توجد الأدلة")


def test_hybrid_retriever_ranks_relevant_document_first():
    retriever = HybridRetriever(documents())
    assert retriever.retrieve("Which metrics include recall and reciprocal rank?")[0].document.doc_id == "D2"
    assert retriever.retrieve("متى يجب الامتناع عن الإجابة؟")[0].document.doc_id == "D3"
