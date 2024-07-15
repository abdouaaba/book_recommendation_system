import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_get_recommendations_without_description(client: TestClient):
    response = client.post("/api/recommend/faiss", json={"num_recommendations": 2})
    assert response.status_code == 422  # the API requires a description

def test_get_recommendations_with_zero(client: TestClient):
    response = client.post("/api/recommend/faiss", json={"description": "A mystery novel", "num_recommendations": 0})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 0

def test_get_recommendations_with_invalid_data_type(client: TestClient):
    response = client.post("/api/recommend/faiss", json={"description": 12345, "num_recommendations": "three"})
    assert response.status_code == 422  # Unprocessable Entity due to invalid data types

def test_get_recommendations_missing_parameters(client: TestClient):
    response = client.post("/api/recommend/faiss")
    assert response.status_code == 422  # Unprocessable Entity due to missing both parameters

@pytest.mark.parametrize("description,num_recommendations", [
    ("", 3),  # Empty description
    ("A science fiction novel", -5),  # Negative number of recommendations
    (None, 3),  # None as description
    ("A science fiction novel", None)  # None as number of recommendations
])
def test_get_recommendations_edge_cases(client: TestClient, description: str, num_recommendations: int):
    response = client.post("/api/recommend/faiss", json={"description": description, "num_recommendations": num_recommendations})
    assert response.status_code == 422  # Unprocessable Entity for all edge cases

def test_get_recommendations(client: TestClient):
    response = client.post("/api/recommend/faiss", json={"description": "A science fiction novel about space exploration", "num_recommendations": 3})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3
    for recommendation in data["recommendations"]:
        assert "title" in recommendation
        assert "description" in recommendation