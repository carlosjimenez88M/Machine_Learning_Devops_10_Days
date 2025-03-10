'''
    Train Model and Save Best Model
    Release Date: 2025-01-27
'''

#=====================#
# ---- Libraries ---- #
#=====================#
import os
import logging
import argparse
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

#=======================================#
# ---- Custom Transformer: Clustering ----#
#=======================================#

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        return self.kmeans.transform(X)

#=========================#
# ---- Sweep Function ----#
#=========================#

def train_model(config=None):
    with wandb.init(job_type="model_fine_tuning", config=config) as run:
        config = wandb.config

        logger.info('Loading data...')
        train_data_path = wandb.use_artifact(config.train_dattrain_data_artifacta_artifact).file()
        df = pd.read_csv(train_data_path, low_memory=False)

        logger.info("Extracting target from dataframe")
        target_column = "median_house_value"
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' is missing from the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info("Setting up preprocessing pipeline")

        num_features = ["longitude", "latitude", "housing_median_age", "total_rooms",
                        "total_bedrooms", "population", "households", "median_income"]


        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cluster", ClusterSimilarity(n_clusters=config.n_clusters, gamma=config.gamma), ["longitude", "latitude"]),
            ]
        )

        logger.info("Splitting train/val")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=config.val_size,
            random_state=config.random_seed,
        )

        logger.info("Applying preprocessing pipeline")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        logger.info("Initializing Random Forest model")
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_seed,
        )

        logger.info("Training the model")
        model.fit(X_train_processed, y_train)

        logger.info("Evaluating the model")
        y_pred = model.predict(X_val_processed)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        wandb.log({"rmse": rmse,
                   "n_estimators": config.n_estimators,
                   "max_depth": config.max_depth,
                   "min_samples_split": config.min_samples_split,
                   "min_samples_leaf": config.min_samples_leaf,
                   "n_clusters": config.n_clusters,
                   "gamma": config.gamma})

        logger.info(f"RMSE on validation set: {rmse:.2f}")

        global best_rmse, best_model_path
        if rmse < best_rmse:
            logger.info("New best model found! Updating...")
            best_rmse = rmse

            # Save the model locally
            model_path = f"best_model.joblib"
            joblib.dump(model, model_path)

            # Log the best model as a W&B artifact
            artifact = wandb.Artifact(
                name="best_random_forest_model",
                type="model",
                description=f"Best Random Forest model with RMSE={rmse:.2f}",
                metadata=dict(config)  # Save the full config as metadata
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            logger.info("Best model saved and registered in W&B.")

        return rmse


#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    # Configuración del sweep
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "rmse", "goal": "minimize"},
        "parameters": {
            "n_estimators": {"min": 10, "max": 500},
            "max_depth": {"min": 5, "max": 50},
            "min_samples_split": {"min": 2, "max": 20},
            "min_samples_leaf": {"min": 1, "max": 20},
            "n_clusters": {"min": 1, "max": 30},
            "gamma": {"min": 0.01, "max": 1.0},
            "val_size": {"value": args.val_size},
            "random_seed": {"value": args.random_seed},
            "train_data_artifact": {"value": args.train_data_artifact},
        },
    }

    logger.info("Creating W&B sweep")
    sweep_id = wandb.sweep(sweep_config, project="random_forest_end_to_end")

    # Initialize global variables to track the best model
    global best_rmse, best_model_path
    best_rmse = float("inf")
    best_model_path = None

    logger.info("Running W&B sweep agent")
    wandb.agent(sweep_id, function=train_model)

    logger.info("Sweep completed. Best model saved at 'best_model.joblib'.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and 6-evaluate a Random Forest model")

    parser.add_argument('--train_data_artifact',
                        type=str,
                        required=True,
                        help="Name of the input data artifact in W&B")

    parser.add_argument('--val_size',
                        type=float,
                        default=0.2,
                        help="Proportion of validation set")

    parser.add_argument('--random_seed',
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    go(args)
