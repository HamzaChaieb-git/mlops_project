import pytest
from fastapi.testclient import TestClient
from src.api.endpoints import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test the prediction endpoint."""
    test_input = {
        "total_day_minutes": 120.0,
        "customer_service_calls": 2,
        "international_plan": "yes",
        "total_intl_minutes": 12.0,
        "total_intl_calls": 3,
        "total_eve_minutes": 220.0,
        "number_vmail_messages": 27,
        "voice_mail_plan": "yes"
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
    assert "model_version" in response.json()
    assert "prediction_time" in response.json()

def test_invalid_predict_input():
    """Test prediction endpoint with invalid input."""
    invalid_input = {
        "total_day_minutes": "invalid",  # Should be float
        "customer_service_calls": 2
    }
    
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "metrics" in response.json()
    assert "timestamp" in response.json()

@pytest.mark.integration
def test_retrain_endpoint():
    """Test the model retraining endpoint."""
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "metrics" in response.json()
    assert response.json()["status"] == "success"