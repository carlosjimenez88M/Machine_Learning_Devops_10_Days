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
import pandas as pd
import numpy as np
import joblib

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

#=========================#
# ---- Sweep Function ----#
#=========================#

def train_model(config=None):
    with wandb.init(job_type="model_fine_tuning",
                    config=config) as run:
        config = wandb.config

        logger.info('Loading data...')
        train_data_path = wandb.use_artifact(config.train_data_artifact).file()
        df = pd.read_csv(train_data_path,
                         low_memory=False)

        logger.info("Extracting target from dataframe")
        target_column = "median_house_value"
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' is missing from the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info("Splitting train/val")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=config.val_size,
            random_state=config.random_seed,
        )

        logger.info("Initializing Random Forest model")
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_seed,
        )

        logger.info("Training the model")
        model.fit(X_train, y_train)

        logger.info("Evaluating the model")
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))


        wandb.log({"rmse": rmse})
        logger.info(f"RMSE on validation set: {rmse:.2f}")


        model_path = f"model_{wandb.run.id}.joblib"
        joblib.dump(model, model_path)
        artifact = wandb.Artifact(
            name=f"random_forest_model_{wandb.run.id}",
            type="model",
            description=f"Random Forest model with RMSE={rmse:.2f}",
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        logger.info("Model saved and registered as artifact")
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
            "val_size": {"value": args.val_size},
            "random_seed": {"value": args.random_seed},
            "train_data_artifact": {"value": args.train_data_artifact},
        },
    }

    logger.info("Creating W&B sweep")
    sweep_id = wandb.sweep(sweep_config, project="random_forest_end_to_end")

    logger.info("Running W&B sweep agent")
    wandb.agent(sweep_id, function=train_model)

    logger.info("Sweep completed.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest model")

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

    parser.add_argument('--sweep_count',
                        type=int,
                        default=20,
                        help="Number of sweep iterations")

    args = parser.parse_args()
    go(args)
