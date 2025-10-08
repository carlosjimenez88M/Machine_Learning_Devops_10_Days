#!/usr/bin/env python
'''
    Predictions Router
    Release Date: 2025-01-27
'''

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
from loggers_configuration import setup_colored_logger
from model_loader import model_loader

logger = setup_colored_logger("PredictionsRouter", "INFO")

router = APIRouter()

class HistoricalDataPoint(BaseModel):
    """Historical data point for computing lag features"""
    time_period: int = Field(..., description="Time period")
    value: float = Field(..., description="Actual value for that period")

class PredictionInput(BaseModel):
    """Input schema for prediction - only base features needed"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    state_encoded: int = Field(..., description="Encoded state value")
    indicator_encoded: int = Field(..., description="Encoded indicator value")
    group_encoded: int = Field(..., description="Encoded group value")
    phase_encoded: int = Field(..., description="Encoded phase value")
    quartile_range: float = Field(..., description="Quartile range value")
    high_ci: float = Field(..., description="High confidence interval")
    low_ci: float = Field(..., description="Low confidence interval")
    quartile_number: float = Field(..., description="Quartile number")
    suppression_flag: int = Field(..., description="Suppression flag (0 or 1)")
    time_period: int = Field(..., description="Time period value")
    historical_values: Optional[List[HistoricalDataPoint]] = Field(
        None,
        description="Historical values for computing lag features (last 3 periods). If not provided, lag features will be set to 0."
    )

class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    model_config = ConfigDict(protected_namespaces=())

    prediction: float = Field(..., description="Predicted value")
    computed_features: dict = Field(..., description="Computed lag and engineered features")
    model_version: str = Field(..., description="Model version used")

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    inputs: List[PredictionInput] = Field(..., description="List of prediction inputs")

class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    model_config = ConfigDict(protected_namespaces=())

    predictions: List[float] = Field(..., description="List of predicted values")
    model_version: str = Field(..., description="Model version used")


def compute_features(input_data: PredictionInput) -> dict:
    """
    Compute lag features and engineered features from input data
    """
    features = {
        'state_encoded': input_data.state_encoded,
        'indicator_encoded': input_data.indicator_encoded,
        'group_encoded': input_data.group_encoded,
        'phase_encoded': input_data.phase_encoded,
        'quartile_range': input_data.quartile_range,
        'high_ci': input_data.high_ci,
        'low_ci': input_data.low_ci,
        'quartile_number': input_data.quartile_number,
        'suppression_flag': input_data.suppression_flag,
        'time_period': input_data.time_period
    }

    # Compute CI width
    features['ci_width'] = input_data.high_ci - input_data.low_ci

    # Compute lag features from historical data
    if input_data.historical_values and len(input_data.historical_values) > 0:
        # Sort historical values by time_period
        hist_df = pd.DataFrame([{
            'time_period': h.time_period,
            'value': h.value
        } for h in input_data.historical_values])
        hist_df = hist_df.sort_values('time_period')

        # Lag features (most recent values)
        if len(hist_df) >= 1:
            features['value_lag_1'] = hist_df.iloc[-1]['value']
        else:
            features['value_lag_1'] = 0.0

        if len(hist_df) >= 2:
            features['value_lag_2'] = hist_df.iloc[-2]['value']
        else:
            features['value_lag_2'] = 0.0

        if len(hist_df) >= 3:
            features['value_lag_3'] = hist_df.iloc[-3]['value']
        else:
            features['value_lag_3'] = 0.0

        # Rolling mean (last 3 periods)
        if len(hist_df) >= 1:
            last_n = min(3, len(hist_df))
            features['value_rolling_mean_3'] = hist_df.tail(last_n)['value'].mean()
        else:
            features['value_rolling_mean_3'] = 0.0

        # Difference (current - previous)
        if len(hist_df) >= 1:
            features['value_diff'] = hist_df.iloc[-1]['value'] - (hist_df.iloc[-2]['value'] if len(hist_df) >= 2 else 0)
        else:
            features['value_diff'] = 0.0

    else:
        # No historical data provided, use zeros
        features['value_lag_1'] = 0.0
        features['value_lag_2'] = 0.0
        features['value_lag_3'] = 0.0
        features['value_rolling_mean_3'] = 0.0
        features['value_diff'] = 0.0

    return features


@router.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(input_data: PredictionInput):
    """
    Make a single prediction with automatic feature engineering
    """
    try:
        logger.info("Received prediction request")

        # Load model
        model = model_loader.get_model()

        # Compute all features including lag features
        logger.info("Computing features...")
        features = compute_features(input_data)

        # Convert to DataFrame with correct feature order
        expected_features = [
            'state_encoded', 'indicator_encoded', 'group_encoded', 'phase_encoded',
            'quartile_range', 'high_ci', 'low_ci', 'quartile_number',
            'suppression_flag', 'time_period', 'value_lag_1', 'value_lag_2',
            'value_lag_3', 'value_rolling_mean_3', 'value_diff', 'ci_width'
        ]

        df = pd.DataFrame([features])[expected_features]

        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(df)[0]

        logger.info(f"Prediction: {prediction}")

        # Extract computed features for transparency
        computed_features = {
            'value_lag_1': features['value_lag_1'],
            'value_lag_2': features['value_lag_2'],
            'value_lag_3': features['value_lag_3'],
            'value_rolling_mean_3': features['value_rolling_mean_3'],
            'value_diff': features['value_diff'],
            'ci_width': features['ci_width']
        }

        return PredictionOutput(
            prediction=float(prediction),
            computed_features=computed_features,
            model_version="random_forest_model:latest"
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionOutput, status_code=status.HTTP_200_OK)
def predict_batch(batch_input: BatchPredictionInput):
    """
    Make batch predictions with automatic feature engineering
    """
    try:
        logger.info(f"Received batch prediction request with {len(batch_input.inputs)} inputs")

        # Handle empty input list
        if len(batch_input.inputs) == 0:
            logger.info("Empty input list, returning empty predictions")
            return BatchPredictionOutput(
                predictions=[],
                model_version="random_forest_model:latest"
            )

        # Load model
        model = model_loader.get_model()

        # Compute features for each input
        logger.info("Computing features for all inputs...")
        features_list = []
        for input_data in batch_input.inputs:
            features = compute_features(input_data)
            features_list.append(features)

        # Convert to DataFrame with correct feature order
        expected_features = [
            'state_encoded', 'indicator_encoded', 'group_encoded', 'phase_encoded',
            'quartile_range', 'high_ci', 'low_ci', 'quartile_number',
            'suppression_flag', 'time_period', 'value_lag_1', 'value_lag_2',
            'value_lag_3', 'value_rolling_mean_3', 'value_diff', 'ci_width'
        ]

        df = pd.DataFrame(features_list)[expected_features]

        # Make predictions
        logger.info("Making batch predictions...")
        predictions = model.predict(df)

        logger.info(f"Batch predictions completed: {len(predictions)} predictions made")

        return BatchPredictionOutput(
            predictions=[float(p) for p in predictions],
            model_version="random_forest_model:latest"
        )

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/model/info", status_code=status.HTTP_200_OK)
def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        logger.info("Model info request received")

        model = model_loader.get_model()

        info = {
            "model_type": type(model).__name__,
            "model_version": "random_forest_model:latest",
            "n_features": model.n_features_in_,
            "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
            "feature_names": [
                'state_encoded', 'indicator_encoded', 'group_encoded', 'phase_encoded',
                'quartile_range', 'high_ci', 'low_ci', 'quartile_number',
                'suppression_flag', 'time_period', 'value_lag_1', 'value_lag_2',
                'value_lag_3', 'value_rolling_mean_3', 'value_diff', 'ci_width'
            ],
            "computed_features": [
                'value_lag_1', 'value_lag_2', 'value_lag_3',
                'value_rolling_mean_3', 'value_diff', 'ci_width'
            ],
            "note": "Lag features are computed automatically from historical_values. Provide last 3 periods for best accuracy."
        }

        logger.info("Model info retrieved successfully")
        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )
