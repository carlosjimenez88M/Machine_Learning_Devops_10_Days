#!/usr/bin/env python
'''
    Trainng model step of the pipeline.
    Release date : 2025-09-07
'''

# =====================#
# ---- libraries ---- #
# =====================#

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from loggers_configuration import setup_colored_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===============================#
# ---- Logger Configuration ----#
# ===============================#

logger = setup_colored_logger("TrainingModel", "INFO")


# =========================#
# ---- main function ---- #
# =========================#

def go(args):
    '''
    Train the model
    '''
    run = None
    try:
        logger.info('Initializing W&B run for model training...')
        run = wandb.init(project='Post-COVID',
                        group='training',
                        job_type="train_model")
        run.config.update(args)

        logger.info(f'Downloading artifact: {args.input_artifact}')
        artifact = run.use_artifact(args.input_artifact)
        artifact_path = artifact.file()

        logger.info('Reading training dataset...')
        df = pd.read_csv(artifact_path)
        logger.info(f'Dataset shape: {df.shape}')

        logger.info('Preparing features and target...')
        feature_columns = [
            'indicator_encoded',
            'group_encoded',
            'state_encoded',
            'subgroup_encoded',
            'time_period',
            'phase',
            'value_lag_1',
            'value_lag_2',
            'value_lag_3',
            'value_rolling_mean_3',
            'value_diff',
            'ci_width',
            'low_ci',
            'high_ci']

        X = df[feature_columns].copy()
        y = df['value'].copy()

        logger.info('Handling missing values in value_diff...')
        X['value_diff'] = X['value_diff'].fillna(0)

        wandb.config.update({
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "features": feature_columns
        })

        logger.info('Creating temporal split...')
        df_model = df.sort_values('time_period').reset_index(drop=True)
        time_periods = sorted(df_model['time_period'].unique())
        train_periods = time_periods[:-3]
        test_periods = time_periods[-3:]
        logger.info(f'   Train periods: {train_periods[0]} to {train_periods[-1]}')
        logger.info(f'   Test periods: {test_periods[0]} to {test_periods[-1]}')

        train_mask = df_model['time_period'].isin(train_periods)
        test_mask = df_model['time_period'].isin(test_periods)

        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[train_mask].copy()
        y_test = y[test_mask].copy()

        logger.info(f'Train set: {len(X_train)} samples')
        logger.info(f'Test set: {len(X_test)} samples')

        wandb.config.update({
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_periods": f"{train_periods[0]}-{train_periods[-1]}",
            "test_periods": f"{test_periods[0]}-{test_periods[-1]}"})

        logger.info('Training RandomForestRegressor model...')
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            bootstrap=args.bootstrap,
            random_state=args.random_state,
            n_jobs=-1
        )

        logger.info('Performing time series cross-validation...')
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(rf,
                                    X_train,
                                    y_train,
                                    cv=tscv,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=-1)
        logger.info(f'   CV MAE scores: {-cv_scores}')
        logger.info(f'   Average CV MAE: {-cv_scores.mean():.4f}')
        cv_mae = -cv_scores.mean()
        wandb.config.update({
            "cv_mae_scores": (-cv_scores).tolist(),
            "cv_mae_mean": cv_mae,
            "cv_mae_std": cv_scores.std()
        })

        logger.info('Fitting model on full training set...')
        rf.fit(X_train, y_train)

        logger.info('Making predictions...')
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        logger.info('Calculating metrics...')
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        logger.info(f'   Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')
        logger.info(f'   Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
        logger.info(f'   Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')

        wandb.log({
            'cv_mae': cv_mae,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'overfitting_mae': train_mae - test_mae,
            'overfitting_rmse': train_rmse - test_rmse
        })

        logger.info('Analyzing feature importances...')
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info('   Top 10 most important features:')
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f'      {row["feature"]}: {row["importance"]:.4f}')

        wandb.log({"feature_importance": wandb.Table(
            dataframe=feature_importance.head(10))})

        logger.info('Creating feature importance plot...')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            feature_importance['feature'].head(10),
            feature_importance['importance'].head(10))
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        ax.invert_yaxis()
        plt.tight_layout()
        wandb.log({"feature_importance_plot": wandb.Image(fig)})
        plt.close(fig)

        logger.info('Saving model...')
        model_path = "random_forest_model.joblib"
        joblib.dump(rf, model_path)
        logger.info(f'   Model saved to: {model_path}')

        logger.info(f'Creating W&B artifact: {args.output_artifact}')
        model_artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description
        )
        model_artifact.add_file(model_path)

        logger.info('Logging model artifact to W&B...')
        run.log_artifact(model_artifact)

        if os.environ.get('WANDB_MODE') != 'offline':
            model_artifact.wait()

        logger.info('Removing local model file...')
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f'   Deleted local file: {model_path}')

        logger.info('Model training completed successfully.')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        if run:
            run.finish()
            logger.info("W&B run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a RandomForestRegressor model")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output model artifact",
        required=True
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output model artifact",
        required=True
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output model artifact",
        required=True
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of trees in the forest",
        default=185
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Maximum depth of the tree",
        default=36
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        help="Minimum number of samples required to split an internal node",
        default=4
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        help="Minimum number of samples required to be at a leaf node",
        default=8
    )
    parser.add_argument(
        "--max_features",
        type=str,
        help="The number of features to consider when looking for the best split",
        default="sqrt")
    parser.add_argument(
        "--bootstrap",
        type=bool,
        help="Whether bootstrap samples are used when building trees",
        default=False
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Random seed for reproducibility",
        default=42
    )

    args = parser.parse_args()
    go(args)
