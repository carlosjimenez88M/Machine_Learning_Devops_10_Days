'''
    Clean and preprocessing dataset
    Release Date: 2025-01-27
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from loggers_configuration import setup_colored_logger
from skimpy import clean_columns
from sklearn.preprocessing import LabelEncoder

#===============================#
# ---- Logger Configuration ----#
#===============================#

logger = setup_colored_logger("PreprocessDataset", "INFO")


#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    ''' Clean and preprocess the dataset '''

    run = None
    try:
        logger.info('Initializing W&B run for data cleaning...')
        run = wandb.init(project='Post-COVID',
                        group='preprocessing',
                        job_type="clean_data")

        logger.info(f'Downloading artifact: {args.input_artifact}')
        artifact = run.use_artifact(args.input_artifact)
        artifact_path = artifact.file()

        logger.info('Reading dataset...')
        df = pd.read_csv(artifact_path)
        logger.info(f'Dataset shape: {df.shape}')

        logger.info('Cleaning column names...')
        df = clean_columns(df)

        logger.info('Validating missing values...')
        fig = df.isna().mean().sort_values(ascending=True).plot(kind='barh')
        plt.title('Percentage of missing values')
        plt.xlabel('Percentage')
        plt.ylabel('Columns')
        wandb.log({"missing_values_plot": wandb.Image(fig)})
        plt.close()

        # Drop unnecessary columns and rows with missing target values
        df = df.drop(columns=['suppression_flag'])
        df = df[df['value'].notna()].copy()

        logger.info('Encoding categorical variables...')
        label_encoders = {}
        categorical_cols = ['indicator', 'group', 'state', 'subgroup']

        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f'   Encoded: {col} ({len(le.classes_)} unique values)')

        logger.info('Sorting dataset by state, indicator, and time_period...')
        df = df.sort_values(['state', 'indicator', 'time_period'])

        logger.info('Creating lag features...')
        for lag in [1, 2, 3]:
            df[f'value_lag_{lag}'] = df.groupby(['state', 'indicator'])['value'].shift(lag)
            logger.info(f'   Created lag feature: value_lag_{lag}')

        logger.info('Creating rolling mean feature...')
        df['value_rolling_mean_3'] = df.groupby(['state', 'indicator'])['value'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        logger.info('Creating difference feature...')
        df['value_diff'] = df.groupby(['state', 'indicator'])['value'].diff()

        logger.info('Creating confidence interval width feature...')
        df['ci_width'] = df['high_ci'] - df['low_ci']

        logger.info('Removing rows with missing lag values...')
        initial_rows = len(df)
        df = df.dropna(subset=['value_lag_1', 'value_lag_2', 'value_lag_3'])
        logger.info(f'   Dropped {initial_rows - len(df)} rows. Final dataset: {len(df)} rows')

        wandb.log({
            "total_rows": initial_rows,
            "final_rows": len(df),
            "dropped_rows": initial_rows - len(df),
            "n_features": len(df.columns)
        })

        logger.info('Saving preprocessed dataset...')
        output_path = args.output_artifact
        df.to_csv(output_path, index=False)
        logger.info(f'   Saved to: {output_path}')

        logger.info(f'Creating W&B artifact: {args.artifact_name}')
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description,
        )

        artifact.add_file(output_path)
        logger.info('Logging artifact to W&B...')
        run.log_artifact(artifact)

        if os.environ.get('WANDB_MODE') != 'offline':
            artifact.wait()

        logger.info('Removing local file...')
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f'   Deleted local file: {output_path}')

        logger.info('Preprocessing completed successfully.')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        if run:
            run.finish()
            logger.info("W&B run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and preprocess data",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name of the output artifact",
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
