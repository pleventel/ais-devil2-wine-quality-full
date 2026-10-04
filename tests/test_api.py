from fastapi.testclient import TestClient

from wine_quality_api import app

client = TestClient(app)


def test_docs_endpoint():
    response = client.get("/docs")

    assert response.status_code == 200
