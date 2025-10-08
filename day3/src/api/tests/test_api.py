#!/usr/bin/env python
'''
    Unit Tests for FastAPI Application
    Release Date: 2025-01-27
'''

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app

client = TestClient(app)

@pytest.fixture
def mock_model():
    """Fixture to mock the ML model"""
    mock = MagicMock()
    mock.predict.return_value = np.array([11.5])
    mock.n_features_in_ = 16
    mock.n_estimators = 185
    return mock

@pytest.fixture
def sample_prediction_input():
    """Sample input for prediction with historical values"""
    return {
        "state_encoded": 1,
        "indicator_encoded": 5,
        "group_encoded": 2,
        "phase_encoded": 1,
        "quartile_range": 2.5,
        "high_ci": 12.5,
        "low_ci": 10.0,
        "quartile_number": 2.0,
        "suppression_flag": 0,
        "time_period": 202104,
        "historical_values": [
            {"time_period": 202101, "value": 10.5},
            {"time_period": 202102, "value": 10.8},
            {"time_period": 202103, "value": 11.2}
        ]
    }

@pytest.fixture
def sample_prediction_input_no_history():
    """Sample input for prediction without historical values"""
    return {
        "state_encoded": 1,
        "indicator_encoded": 5,
        "group_encoded": 2,
        "phase_encoded": 1,
        "quartile_range": 2.5,
        "high_ci": 12.5,
        "low_ci": 10.0,
        "quartile_number": 2.0,
        "suppression_flag": 0,
        "time_period": 202104
    }

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Post-COVID Predictions API"
    assert "version" in data
    assert data["version"] == "1.0.0"

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@patch('routers.predictions.model_loader.get_model')
def test_predict_endpoint_with_history(mock_get_model, mock_model, sample_prediction_input):
    """Test single prediction endpoint with historical values"""
    mock_get_model.return_value = mock_model

    response = client.post("/api/v1/predict", json=sample_prediction_input)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert "computed_features" in data
    assert isinstance(data["prediction"], float)
    assert data["model_version"] == "random_forest_model:latest"

    # Check computed features
    computed = data["computed_features"]
    assert "value_lag_1" in computed
    assert "value_lag_2" in computed
    assert "value_lag_3" in computed
    assert "value_rolling_mean_3" in computed
    assert "value_diff" in computed
    assert "ci_width" in computed

    # Verify lag values are computed correctly
    assert computed["value_lag_1"] == 11.2  # Most recent
    assert computed["value_lag_2"] == 10.8
    assert computed["value_lag_3"] == 10.5
    assert computed["ci_width"] == 2.5  # high_ci - low_ci

    mock_model.predict.assert_called_once()

@patch('routers.predictions.model_loader.get_model')
def test_predict_endpoint_without_history(mock_get_model, mock_model, sample_prediction_input_no_history):
    """Test prediction endpoint without historical values (zeros for lags)"""
    mock_get_model.return_value = mock_model

    response = client.post("/api/v1/predict", json=sample_prediction_input_no_history)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "computed_features" in data

    # Check that lag features are set to 0
    computed = data["computed_features"]
    assert computed["value_lag_1"] == 0.0
    assert computed["value_lag_2"] == 0.0
    assert computed["value_lag_3"] == 0.0
    assert computed["value_rolling_mean_3"] == 0.0
    assert computed["value_diff"] == 0.0
    assert computed["ci_width"] == 2.5

    mock_model.predict.assert_called_once()

@patch('routers.predictions.model_loader.get_model')
def test_batch_predict_endpoint(mock_get_model, mock_model, sample_prediction_input):
    """Test batch prediction endpoint"""
    mock_get_model.return_value = mock_model
    mock_model.predict.return_value = np.array([11.5, 12.3, 10.8])

    batch_input = {
        "inputs": [
            sample_prediction_input,
            sample_prediction_input,
            sample_prediction_input
        ]
    }

    response = client.post("/api/v1/predict/batch", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "model_version" in data
    assert len(data["predictions"]) == 3
    assert all(isinstance(p, float) for p in data["predictions"])
    mock_model.predict.assert_called_once()

@patch('routers.predictions.model_loader.get_model')
def test_model_info_endpoint(mock_get_model, mock_model):
    """Test model info endpoint"""
    mock_get_model.return_value = mock_model

    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "n_features" in data
    assert "n_estimators" in data
    assert "feature_names" in data
    assert "computed_features" in data
    assert "note" in data
    assert data["n_features"] == 16
    assert data["n_estimators"] == 185
    assert len(data["feature_names"]) == 16
    assert len(data["computed_features"]) == 6

def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing required fields"""
    incomplete_input = {
        "state_encoded": 1,
        "indicator_encoded": 5
        # Missing required fields
    }

    response = client.post("/api/v1/predict", json=incomplete_input)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_invalid_types():
    """Test prediction endpoint with invalid field types"""
    invalid_input = {
        "state_encoded": "invalid",  # Should be int
        "indicator_encoded": 5,
        "group_encoded": 2,
        "phase_encoded": 1,
        "quartile_range": 2.5,
        "high_ci": 12.5,
        "low_ci": 10.0,
        "quartile_number": 2.0,
        "suppression_flag": 0,
        "time_period": 202101
    }

    response = client.post("/api/v1/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error

@patch('routers.predictions.model_loader.get_model')
def test_predict_endpoint_model_error(mock_get_model, sample_prediction_input):
    """Test prediction endpoint when model raises error"""
    mock_get_model.side_effect = Exception("Model loading failed")

    response = client.post("/api/v1/predict", json=sample_prediction_input)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data

@patch('routers.predictions.model_loader.get_model')
def test_batch_predict_empty_list(mock_get_model, mock_model):
    """Test batch prediction with empty list"""
    mock_get_model.return_value = mock_model
    mock_model.predict.return_value = np.array([])

    batch_input = {"inputs": []}

    response = client.post("/api/v1/predict/batch", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 0

@patch('routers.predictions.model_loader.get_model')
def test_compute_features_with_partial_history(mock_get_model, mock_model):
    """Test feature computation with only 2 historical values"""
    mock_get_model.return_value = mock_model

    input_data = {
        "state_encoded": 1,
        "indicator_encoded": 5,
        "group_encoded": 2,
        "phase_encoded": 1,
        "quartile_range": 2.5,
        "high_ci": 12.5,
        "low_ci": 10.0,
        "quartile_number": 2.0,
        "suppression_flag": 0,
        "time_period": 202103,
        "historical_values": [
            {"time_period": 202101, "value": 10.5},
            {"time_period": 202102, "value": 10.8}
        ]
    }

    response = client.post("/api/v1/predict", json=input_data)
    assert response.status_code == 200
    data = response.json()
    computed = data["computed_features"]

    # Should have lag_1 and lag_2, but lag_3 should be 0
    assert computed["value_lag_1"] == 10.8
    assert computed["value_lag_2"] == 10.5
    assert computed["value_lag_3"] == 0.0
