import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_live_predict_endpoint_fallback():
    # Make sure we hit the endpoint using fallback mechanics (or database fallback)
    response = client.post(
        "/predict",
        json={"video_id": 9999, "model_type": "multimodal"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction_score" in data
    assert "is_fake" in data
    assert isinstance(data["prediction_score"], float)
    assert isinstance(data["is_fake"], bool)
    assert "explanation" in data
    assert "multimodal_fusion" in data["explanation"]
