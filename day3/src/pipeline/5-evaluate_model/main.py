#!/usr/bin/env python
'''
    Evaluate Model
    Release Date: 2025-01-27
'''

#=====================#
# ---- Libraries ---- #
#=====================#
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from loggers_configuration import setup_colored_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#===============================#
# ---- Logger Configuration ----#
#===============================#
logger = setup_colored_logger("EvaluateModel", "INFO")


#=========================#
# ---- Main function ---- #
#=========================#

def go(args):
    '''
    Evaluate the trained model on test data
    '''
    run = None
    try:
        logger.info('Initializing W&B run for model evaluation...')
        run = wandb.init(project='Post-COVID',
                        group='evaluation',
                        job_type="evaluate")
        run.config.update(args)

        logger.info(f'Downloading test data artifact: {args.test_data}')
        test_artifact = run.use_artifact(args.test_data)
        test_data_path = test_artifact.file()

        logger.info('Reading test dataset...')
        df_test = pd.read_csv(test_data_path)
        logger.info(f'Test dataset shape: {df_test.shape}')

        logger.info('Extracting features and target from test data...')
        X_test = df_test.drop(columns=['value']).copy()
        y_test = df_test['value'].copy()
        logger.info(f'Test features shape: {X_test.shape}')
        logger.info(f'Test target shape: {y_test.shape}')

        logger.info(f'Downloading model artifact: {args.model_export}')
        model_artifact = run.use_artifact(args.model_export)
        model_path = model_artifact.download()

        logger.info('Loading trained model...')
        model_file = os.path.join(model_path, 'random_forest_model.joblib')
        model = joblib.load(model_file)
        logger.info(f'Model successfully loaded from {model_file}')

        logger.info('Making predictions on test data...')
        y_pred = model.predict(X_test)

        logger.info('Calculating evaluation metrics...')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f'   Test RMSE: {rmse:.4f}')
        logger.info(f'   Test MAE: {mae:.4f}')
        logger.info(f'   Test R²: {r2:.4f}')

        run.summary["test_rmse"] = rmse
        run.summary["test_mae"] = mae
        run.summary["test_r2"] = r2

        wandb.log({
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2
        })

        logger.info('Creating prediction vs actual scatter plot...')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predicted vs Actual Values', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"predicted_vs_actual": wandb.Image(fig)})
        plt.close(fig)

        logger.info('Creating residuals plot...')
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"residuals_plot": wandb.Image(fig)})
        plt.close(fig)

        logger.info('Creating residuals distribution plot...')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"residuals_distribution": wandb.Image(fig)})
        plt.close(fig)

        logger.info('Saving predictions to CSV...')
        results = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'residual': residuals
        })
        results_path = 'test_predictions.csv'
        results.to_csv(results_path, index=False)
        logger.info(f'   Saved predictions to: {results_path}')

        logger.info('Creating W&B artifact for predictions...')
        predictions_artifact = wandb.Artifact(
            name='test_predictions',
            type='predictions',
            description='Test set predictions and residuals'
        )
        predictions_artifact.add_file(results_path)
        run.log_artifact(predictions_artifact)

        if os.environ.get('WANDB_MODE') != 'offline':
            predictions_artifact.wait()

        logger.info('Removing local predictions file...')
        if os.path.exists(results_path):
            os.remove(results_path)
            logger.info(f'   Deleted local file: {results_path}')

        logger.info('Model evaluation completed successfully.')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        if run:
            run.finish()
            logger.info("W&B run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test data",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully qualified name for the test dataset artifact",
        required=True
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully qualified name for the exported model artifact",
        required=True
    )

    args = parser.parse_args()
    go(args)
