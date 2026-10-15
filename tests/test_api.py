from fastapi.testclient import TestClient

from api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_endpoint():
    response = TestClient(app).post(
        "/answer",
        json={"question": "What does retrieval augmented generation combine?", "top_k": 3},
    )
    assert response.status_code == 200
    assert "answer" in response.json()
