'''
    Train Model and Save Best Model
    Release Date: 2025-03-02
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import os
import logging
import argparse
import wandb
import pandas as pd
import numpy as np
import joblib
import mlflow
import tempfile
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

#==========================#
#   Logger Configuration   #
#==========================#

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger()


#=====================#
#    Main Function    #
#=====================#

def go(args):
    run = wandb.init(job_type="train")


    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data_artifact).file()
    df = pd.read_csv(train_data_path, low_memory=False)


    logger.info("Extracting target from dataframe")
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]


    logger.info("Loading model configuration")
    model_config_path = os.path.abspath(args.model_config)


    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)


    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed
    )


    logger.info("Initializing Random Forest model")
    model = RandomForestRegressor(**model_config["random_forest"])


    logger.info("Training the model")
    model.fit(X_train, y_train)


    logger.info("Evaluating the model")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    run.summary["rmse"] = rmse

    logger.info(f"RMSE on validation set: {rmse:.2f}")


    model_path = "best_model.joblib"
    joblib.dump(model, model_path)

    artifact = wandb.Artifact(
        name=args.export_artifact,
        type="model",
        description=f"Random Forest model with RMSE={rmse:.2f}",
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    logger.info("Logging model to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_model_path = os.path.join(temp_dir, "mlflow_model")

        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.save_model(
            model,
            mlflow_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val[:2],
        )

        mlflow.log_artifact(mlflow_model_path, "model")
        logger.info("Model registered in MLflow.")

    logger.info("Best model saved and registered in W&B and MLflow.")


#=====================#
#   Argument Parser  #
#=====================#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model with W&B and MLflow tracking."
    )

    parser.add_argument(
        "--train_data_artifact",
        type=str,
        required=True,
        help="Path to the training data artifact.",
    )

    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file.",
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        required=True,
        help="Name of the artifact for the exported model.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Size for the validation set as a fraction of the training set.",
    )

    args = parser.parse_args()

    go(args)
