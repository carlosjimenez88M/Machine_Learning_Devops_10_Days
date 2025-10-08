#!/usr/bin/env python
'''
    Model Loader for W&B Artifacts
    Release Date: 2025-01-27
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import wandb
from loggers_configuration import setup_colored_logger

logger = setup_colored_logger("ModelLoader", "INFO")

class ModelLoader:
    """Singleton class to load and cache the model"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, artifact_name: str = None):
        """Load model from W&B artifact"""
        if self._model is not None:
            logger.info("Returning cached model")
            return self._model

        try:
            if artifact_name is None:
                artifact_name = "danieljimenez88m-carlosdanieljimenez-com/Post-COVID/random_forest_model:latest"

            logger.info(f"Loading model from W&B artifact: {artifact_name}")

            # Set W&B to offline mode if not authenticated
            if not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_MODE"] = "offline"

            # Initialize W&B run
            run = wandb.init(project="Post-COVID", job_type="inference")

            # Download model artifact
            logger.info("Downloading model artifact...")
            model_artifact = run.use_artifact(artifact_name)
            model_path = model_artifact.download()

            # Load model
            model_file = os.path.join(model_path, "random_forest_model.joblib")
            logger.info(f"Loading model from: {model_file}")
            self._model = joblib.load(model_file)

            logger.info("Model loaded successfully")
            run.finish()

            return self._model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_model(self):
        """Get cached model or load if not cached"""
        if self._model is None:
            return self.load_model()
        return self._model

# Global model loader instance
model_loader = ModelLoader()
