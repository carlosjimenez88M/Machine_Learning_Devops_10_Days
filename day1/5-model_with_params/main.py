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


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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

    # --- INICIO DE LA CONSTRUCCIÓN DE LA PIPELINE ---
    logger.info("Building preprocessing pipeline and model pipeline")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    
    rf_model = RandomForestRegressor(**model_config["random_forest"])

    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('random_forest', rf_model)
    ])
    


    logger.info("Training the full pipeline")
    full_pipeline.fit(X_train, y_train) 


    logger.info("Evaluating the model")
    y_pred = full_pipeline.predict(X_val) 
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    run.summary["rmse"] = rmse

    logger.info(f"RMSE on validation set: {rmse:.2f}")


    logger.info("Saving model locally in MLflow format and logging to W&B...")
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_model_local_path = os.path.join(temp_dir, "random_forest_mlflow_model")


        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.save_model(
            sk_model=full_pipeline, 
            path=mlflow_model_local_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val[:2],
        )

        artifact = wandb.Artifact(
            name=args.export_artifact,
            type="model",
            description=f"Random Forest Pipeline (MLflow format) with RMSE={rmse:.2f}",
        )
        artifact.add_dir(mlflow_model_local_path) # <-- Añade el directorio completo de la Pipeline
        run.log_artifact(artifact)
        logger.info("Model registered in W&B as MLflow structure.")

    logger.info("Best model saved and registered in W&B.")


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