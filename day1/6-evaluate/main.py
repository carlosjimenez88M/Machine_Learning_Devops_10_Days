'''
    Evaluate Model
    Release Date: 2025-03-02
'''


#=====================#
# ---- Libraries ---- #
#=====================#
import os
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
#==========================#
#   Logger Configuration   #
#==========================#
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    """
    Executes a machine learning workflow using data stored as artifacts and employs a trained
    MLflow model for predictions and evaluation. The workflow comprises several steps: downloading
    and reading test data, extracting targets from the test dataset, loading a pre-trained model,
    processing the test dataset, generating predictions, evaluating model performance via RMSE, and
    creating visualizations for analysis. The generated visualizations and results are logged back
    to the WandB dashboard.

    :param args: Arguments with the following attributes:
        - test_data (str): The name of the test dataset artifact.
        - model_export (str): The name of the exported MLflow model artifact.

    :return: None
    """
    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path,
                     low_memory=False)

    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("median_house_value")

    logger.info("Downloading and reading the exported model")

    model_download_root_path = run.use_artifact(args.model_export).download()


    model = mlflow.sklearn.load_model(model_download_root_path)



    logger.info("Transform data with Pipeline")



    preprocessor = model["preprocessor"]


    used_columns = []
    for name, transformer, features in preprocessor.transformers:
        if features is not None:
            used_columns.extend(features)


    X_test_processed = preprocessor.transform(X_test[used_columns])

    logger.info("Making predictions")
    y_pred = model["random_forest"].predict(X_test_processed)

    logger.info("Scoring (RMSE Calculation)")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    run.summary["RMSE"] = rmse
    logger.info(f"RMSE on test set: {rmse:.2f}")

    # ========================= #
    #   Visualization Plots     #
    # ========================= #

    logger.info("Generating Prediction vs Actual Values Scatter Plot")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    ax_scatter.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Prediction")
    ax_scatter.set_xlabel("Actual Values")
    ax_scatter.set_ylabel("Predicted Values")
    ax_scatter.set_title("Prediction vs Actual Values")
    ax_scatter.legend()
    fig_scatter.tight_layout()

    logger.info("Generating Residuals Plot")
    fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
    residuals = y_test - y_pred
    ax_residuals.scatter(y_test, residuals, alpha=0.5)
    ax_residuals.axhline(y=0, color="r", linestyle="--")
    ax_residuals.set_xlabel("Actual Values")
    ax_residuals.set_ylabel("Residuals")
    ax_residuals.set_title("Residuals Plot")
    fig_residuals.tight_layout()


    run.log({
        "prediction_vs_actual": wandb.Image(fig_scatter),
        "residuals_plot": wandb.Image(fig_residuals)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided Random Forest regression model",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)