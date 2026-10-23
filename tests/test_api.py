from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_score_transaction():
    payload = {"features": [0.0] * 29}
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["decision"] in ["approve", "review", "block"]
    assert "model_version" in data