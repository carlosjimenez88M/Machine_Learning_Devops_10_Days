'''
    Split into Train and Test Sets
    Release Date: 2025-01-27
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

#===============================#
# ---- Logger Configuration ----#
#===============================#

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
    """
    Splits the input data into training and testing sets, saves the splits as
    temporary CSV files, logs them as artifacts using `wandb`, and preserves
    the target column.

    The function begins by extracting the input data artifact using WandB
    and loading it into a DataFrame. The target column is separated, and the
    remaining data is split into training and testing sets proportionally
    based on the specified test size. The resulting splits are saved as CSV
    files in a temporary directory. Each split is then logged as a separate
    artifact with the metadata provided in the arguments.

    :param args: The command-line arguments containing parameters for running
        the train/test split process.
    :type args: argparse.Namespace
    :return: None
    """
    logger.info("Initializing train/test split...")
    run = wandb.init(job_type="split_data")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    logger.info("Separating target column...")
    target_column = "median_house_value"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    train_data = X_train.copy()
    train_data[target_column] = y_train

    test_data = X_test.copy()
    test_data[target_column] = y_test

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split, data in {"train": train_data, "test": test_data}.items():
            artifact_name = f"{args.artifact_root}_{split}.csv"
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Saving {split} dataset to {artifact_name}")
            data.to_csv(temp_path, index=False)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info(f"Logging {split} dataset as artifact")
            run.log_artifact(artifact)
            artifact.wait()

    logger.info("Train/test split completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the produced artifacts", required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    args = parser.parse_args()

    go(args)
