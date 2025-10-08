# Post-COVID Predictions API

FastAPI application for serving predictions from the trained Random Forest model with automatic feature engineering.

## Requirements

- **Python 3.11** (required)
- pip
- Virtual environment (created automatically)

## Quick Start

### 1. Setup and Install (Python 3.11)

```bash
cd /Users/carlosdaniel/Documents/Projects/Personal_Projects/Machine_Learning_Devops_10_Days/day3/src/api

# Create venv with Python 3.11 and install dependencies
make install
```

### 2. Run Tests

```bash
make test
```

### 3. Start API Server

```bash
make run
```

API will be available at: `http://localhost:8000`

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Create Python 3.11 virtual environment |
| `make install` | Install all dependencies (includes setup) |
| `make test` | Run unit tests |
| `make run` | Start FastAPI server on port 8000 |
| `make clean` | Remove venv and cache files |

## Features

- **Automatic Feature Engineering**: Computes lag features, rolling means, and differences automatically
- **Historical Data Support**: Provide last 3 periods for accurate predictions
- **Model Caching**: Model loaded once and cached for performance
- **Batch Predictions**: Process multiple predictions in a single request
- **Input Validation**: Automatic validation using Pydantic
- **Unit Tests**: Full test coverage (11/11 tests passing)

## API Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Single Prediction
```bash
POST /api/v1/predict
```

**Request:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 11.35,
  "computed_features": {
    "value_lag_1": 11.2,
    "value_lag_2": 10.8,
    "value_lag_3": 10.5,
    "value_rolling_mean_3": 10.83,
    "value_diff": 0.4,
    "ci_width": 2.5
  },
  "model_version": "random_forest_model:latest"
}
```

### Batch Prediction
```bash
POST /api/v1/predict/batch
```

**Request:**
```json
{
  "inputs": [
    { "state_encoded": 1, ... },
    { "state_encoded": 2, ... }
  ]
}
```

### Model Info
```bash
GET /api/v1/model/info
```

Returns model metadata and feature information.

## Feature Engineering

### Input Features (required)
- Base features: state, indicator, group, phase, quartile, CI values, time period
- Historical values (optional): Last 3 periods for lag computation

### Computed Features (automatic)
- **value_lag_1, value_lag_2, value_lag_3**: Lag features from historical data
- **value_rolling_mean_3**: Rolling mean of last 3 periods
- **value_diff**: Difference from previous period
- **ci_width**: Calculated as `high_ci - low_ci`

**Note**: If no historical values provided, lag features default to 0.

## Testing

Run the full test suite:
```bash
make test
```

Test coverage:
- ✅ Root and health endpoints
- ✅ Single predictions with/without history
- ✅ Batch predictions
- ✅ Model info
- ✅ Input validation
- ✅ Error handling
- ✅ Feature computation with partial history

## Example Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
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
)

print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Clean Up

Remove virtual environment and cache:
```bash
make clean
```

## Troubleshooting

### Python 3.11 not found
Install Python 3.11:
```bash
brew install python@3.11  # macOS
```

### Module not found errors
Reinstall dependencies:
```bash
make clean
make install
```

## Architecture

```
api/
├── main.py                 # FastAPI application
├── model_loader.py         # Model loader with caching
├── routers/
│   └── predictions.py      # Endpoints with feature engineering
├── tests/
│   └── test_api.py        # Unit tests (11/11 passing)
├── Makefile               # Build automation
├── requirements.txt       # Python 3.11 dependencies
└── README.md             # This file
```

## Dependencies

- FastAPI 0.115.12 - Web framework
- Uvicorn 0.34.4 - ASGI server
- Pydantic 2.10.5 - Data validation
- Pandas 2.2.0 - Data manipulation
- Scikit-learn 1.4.0 - ML models
- W&B 0.16.3 - Model artifacts
- Pytest 8.3.4 - Testing

All versions are compatible with Python 3.11.
