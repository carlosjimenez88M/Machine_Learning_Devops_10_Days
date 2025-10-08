# Day 3: Post-COVID ML System - API & Experiments

Complete ML system with FastAPI serving, MLflow tracking, and hyperparameter optimization for Post-COVID predictions.

## 🚀 Quick Start

### API (FastAPI Prediction Service)
```bash
cd src/api
make install    # Setup Python 3.11 venv
make test       # Run tests (11/11 passing)
make run        # Start server → http://localhost:8000
```

### Experiments (Hyperparameter Optimization)
```bash
cd src/experiments
make install         # Setup Python 3.11 venv
make run             # Run pipeline
make sweep COUNT=20  # Hyperparameter optimization
make save-params     # Save best params to JSON
make register        # Register to MLflow
```

## 📁 Project Structure

```
day3/
├── src/
│   ├── api/                    # FastAPI service
│   │   ├── Makefile           ← API commands
│   │   └── README.md
│   ├── experiments/            # MLflow experiments
│   │   ├── Makefile           ← Experiments commands
│   │   └── README.md
│   ├── pipeline/              # Original pipeline
│   └── loggers_configuration/ # Shared logging
└── README.md                  # This file
```

## 📖 Documentation

- **[API Documentation](src/api/README.md)** - FastAPI endpoints, testing, deployment
- **[Experiments Documentation](src/experiments/README.md)** - Hyperparameter optimization, MLflow tracking

## ✨ Key Features

### API
- ✅ Automatic lag feature computation
- ✅ Batch predictions
- ✅ 11/11 unit tests passing
- ✅ Interactive docs (Swagger/ReDoc)
- ✅ Python 3.11

### Experiments
- ✅ Bayesian hyperparameter optimization
- ✅ MLflow experiment tracking
- ✅ W&B sweep integration
- ✅ Model registry
- ✅ JSON export of best params
- ✅ Python 3.11

## 🛠 Requirements

- Python 3.11
- W&B account
- Kaggle credentials

## 📊 Commands Summary

| Component | Command | Description |
|-----------|---------|-------------|
| **API** | `make install` | Setup Python 3.11 environment |
| | `make test` | Run unit tests |
| | `make run` | Start API server |
| **Experiments** | `make install` | Setup Python 3.11 environment |
| | `make run` | Run pipeline |
| | `make sweep` | Hyperparameter optimization |
| | `make save-params` | Export best parameters |
| | `make register` | Register best model |

See component READMEs for detailed documentation.
