'''
    Clean and preprocessing dataset
    In this script, the database will only be organized and cleaned.
    Release Date: 2025-01-26
'''

#=====================#
# ---- libraries ---- #
#=====================#

import os
import argparse
import logging
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    """
    Executes the data cleaning and preprocessing pipeline. This function initializes
    a Weights & Biases (W&B) run to download an input dataset artifact, processes
    the dataset to handle missing values, stratifies it based on income categories,
    and combines the stratified train-test split into a single dataset. The processed
    data is then saved, logged back as an artifact in W&B, and the temporary file
    is removed.

    :param args: Arguments necessary for the function, which include:
        - input_artifact (str): Name of the input artifact to download from W&B.
        - n_splits (int): Number of splits for `StratifiedShuffleSplit`.
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Random seed for shuffling and stratification.
        - artifact_name (str): Name of the output artifact to be logged into W&B.
        - artifact_type (str): Type of the artifact to be tagged in W&B.
        - artifact_description (str): Description of the processed artifact to log.

    :return: None
    """
    logger.info('Initializing clean data process...')
    run = wandb.init(job_type="clean_data")
    logger.info('Downloading dataset...')
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)


    logger.info('Checking for missing values...')
    logger.info(f"Missing values per column:\n{df.isnull().sum()}")


    logger.info("Creating income categories for stratification...")
    df["income_cat"] = pd.cut(df["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    splitter = StratifiedShuffleSplit(n_splits=args.n_splits,
                                      test_size=args.test_size,
                                      random_state=args.random_state)
    for train_index, test_index in splitter.split(df, df["income_cat"]):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]
        break


    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    combined_data = pd.concat([strat_train_set, strat_test_set], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=args.random_state).reset_index(drop=True)


    logger.info("Saving combined dataset...")
    combined_output_path = "combined_data.csv"
    combined_data.to_csv(combined_output_path, index=False)


    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(combined_output_path)

    logger.info("Logging artifact")
    run.log_artifact(artifact)


    os.remove(combined_output_path)
    logger.info("Clean data process completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean data")

    parser.add_argument('--input_artifact',
                        type=str,
                        required=True,
                        help="Input file path")

    parser.add_argument('--artifact_name',
                        type=str,
                        required=True,
                        help="Name of the output artifact")

    parser.add_argument('--artifact_description',
                        type=str,
                        help="Description of the output artifact",
                        required=True)

    parser.add_argument('--artifact_type',
                        type=str,
                        required=True,
                        help="Type of the output artifact")

    parser.add_argument('--n_splits',
                        type=int,
                        help="Number of splits for StratifiedShuffleSplit",
                        required=True)

    parser.add_argument('--test_size',
                        type=float,
                        help="Test size for StratifiedShuffleSplit",
                        required=True)

    parser.add_argument('--random_state',
                        type=int,
                        help="Random state for StratifiedShuffleSplit",
                        required=True)

    args = parser.parse_args()
    go(args)
