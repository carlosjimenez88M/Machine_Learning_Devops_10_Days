'''
            Step #1 Download Dataset
    Description: Download Kaggle dataset and version control
    Release Date: 2025-10-06
'''



#=====================#
# ---- libraries ---- #
#=====================#

import kagglehub
import wandb
import pandas as pd
import argparse
import os

from loggers_configuration import setup_colored_logger

#===============================#
# ---- Logger Configuration ----#
#===============================#

logger = setup_colored_logger("DownloadDataset", "INFO")


#========================#
# ---- main program ---- #
#========================#


def go(args):
    ''' Download Kaggle dataset and version control '''

    run = None
    try:
        logger.info("Initializing a new W&B run...")
        run = wandb.init(project='Post-COVID',
                         group='download_data',
                         job_type='download_data')


        logger.info(f"Downloading dataset '{args.kaggle_dataset}' from KaggleHub...")
        path = kagglehub.dataset_download(args.kaggle_dataset)


        file_to_upload = os.path.join(path, args.file)
        logger.info(f"File downloaded to: {file_to_upload}")


        logger.info(f"Creating W&B artifact: '{args.artifact_name}'")
        artifact = wandb.Artifact(
                    name=args.artifact_name,
                    type=args.artifact_type,
                    description=args.artifact_description
        )


        artifact.add_file(file_to_upload, name=args.file)

        logger.info("Logging artifact to W&B...")
        run.log_artifact(artifact)

        if os.environ.get('WANDB_MODE') != 'offline':
            artifact.wait()

        logger.info("Artifact logged successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


    finally:
        if run:
            run.finish()
            logger.info("W&B run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle Dataset")

    parser.add_argument(
        "--file",
        type=str,
        help="Name of data file to be uploaded as artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name of the artifact to be logged",
        required=True
    )
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        help="Kaggle dataset identifier",
        required=False,
        default="programmerrdai/post-covid-conditions"
    )
    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the artifact to be logged",
        required=True
    )
    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description of the artifact to be logged",
        required=True
    )
    parser.add_argument(
        "--artifact_metadata",
        type=str,
        help="Metadata of the artifact to be logged in key1=value1,key2=value2 format",
        required=False,
        default=""
    )
    args = parser.parse_args()
    go(args)

