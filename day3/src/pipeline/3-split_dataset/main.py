'''
    Split into Train and Test Sets
    Release Date: 2025-01-27
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import argparse
import os

import pandas as pd
import wandb

from loggers_configuration import setup_colored_logger

#===============================#
# ---- Logger Configuration ----#
#===============================#

logger = setup_colored_logger("SplitDataset", "INFO")


#=========================#
# ---- Main Function ---- #
#=========================#
def go(args):
    logger.info('Splitting data into train and test sets')
    run = wandb.init(job_type="split_data")
    run.config.update(args)

    logger.info('Downloading and reading artifact')
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    logger.info('Splitting data')
    df_model = df.dropna(subset=['value_lag_1', 
                                 'value_lag_2', 
                                 'value_lag_3']).copy()
    
    feature_columns = [
    'indicator_encoded', 
    'group_encoded', 
    'state_encoded',
    'subgroup_encoded',
    'time_period', 'phase',
    'value_lag_1', 'value_lag_2', 'value_lag_3',
    'value_rolling_mean_3', 'value_diff', 'ci_width',
    'low_ci', 'high_ci'
    ]

    X = df_model[feature_columns].copy()
    y = df_model['value'].copy()

    # Rellenar NaN en value_diff
    X['value_diff'] = X['value_diff'].fillna(0)

    wandb.config.update({
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "features": feature_columns
    })
    logger.info(f"Features used for modeling: {feature_columns}")
    df_model = df_model.sort_values('time_period')
    time_periods = sorted(df_model['time_period'].unique())
    train_periods = time_periods[:-3]
    test_periods = time_periods[-3:]

    train_mask = df_model['time_period'].isin(train_periods)
    test_mask = df_model['time_period'].isin(test_periods)

    X_train = X[train_mask].reset_index(drop=True).copy()
    X_test = X[test_mask].reset_index(drop=True).copy()
    y_train = y[train_mask].reset_index(drop=True).copy()
    y_test = y[test_mask].reset_index(drop=True).copy()
    
    
    wandb.config.update({
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_periods": f"{train_periods[0]}-{train_periods[-1]}",
        "test_periods": f"{test_periods[0]}-{test_periods[-1]}"
    })

    # Guardar train data
    logger.info('Saving train dataset...')
    train_data = X_train.copy()
    train_data['value'] = y_train.values
    train_data.to_csv(args.train_artifact, index=False)
    logger.info(f'   Train data saved to: {args.train_artifact}')

    # Guardar test data
    logger.info('Saving test dataset...')
    test_data = X_test.copy()
    test_data['value'] = y_test.values
    test_data.to_csv(args.test_artifact, index=False)
    logger.info(f'   Test data saved to: {args.test_artifact}')

    # Crear y subir artifact de train
    logger.info(f'Creating W&B artifact for train data: {args.train_artifact}')
    train_artifact = wandb.Artifact(
        name=args.train_artifact,
        type=args.artifact_type,
        description=f"{args.artifact_description} - Train set"
    )
    train_artifact.add_file(args.train_artifact)
    logger.info('Logging train artifact to W&B...')
    run.log_artifact(train_artifact)

    if os.environ.get('WANDB_MODE') != 'offline':
        train_artifact.wait()

    # Crear y subir artifact de test
    logger.info(f'Creating W&B artifact for test data: {args.test_artifact}')
    test_artifact = wandb.Artifact(
        name=args.test_artifact,
        type=args.artifact_type,
        description=f"{args.artifact_description} - Test set"
    )
    test_artifact.add_file(args.test_artifact)
    logger.info('Logging test artifact to W&B...')
    run.log_artifact(test_artifact)

    if os.environ.get('WANDB_MODE') != 'offline':
        test_artifact.wait()

    # Limpiar archivos locales
    logger.info('Removing local files...')
    if os.path.exists(args.train_artifact):
        os.remove(args.train_artifact)
        logger.info(f'   Deleted local file: {args.train_artifact}')
    if os.path.exists(args.test_artifact):
        os.remove(args.test_artifact)
        logger.info(f'   Deleted local file: {args.test_artifact}')

    logger.info('Split completed successfully.')
    run.finish()
    logger.info("W&B run finished.")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--train_artifact",
        type=str,
        help="Name of the train artifact",
        required=True
    )   

    parser.add_argument(
        "--test_artifact",
        type=str,       
        help="Name of the test artifact",
        required=True
    )
    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )
    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    args = parser.parse_args()
    go(args)
