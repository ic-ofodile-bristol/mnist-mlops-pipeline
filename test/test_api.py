import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:   # the `with` triggers lifespan -> model loads
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_happy_path(client):
    with open("samples/sample_0_label7.png", "rb") as f:
        response = client.post("/predict", files={"file": ("digit.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 7
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 400