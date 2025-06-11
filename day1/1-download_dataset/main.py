'''
            Step #1 Download Dataset
    Description: Program for generating the download
    of databases for the pipeline with Weights and Biases.
    Release Date: 2025-01-26
'''

# =====================#
# ---- Libraries ---- #
# =====================#

import argparse
import logging
import pathlib
import tempfile

import requests
import wandb

# ================================#
# ---- Logger Configuration ---- #
# ================================#

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# =========================#
# ---- Main Function ---- #
# =========================#


def go(args):
    """
    Downloads a file from a given URL, creates a "wandb" artifact, and logs it
    into the "wandb" system.

    The function retrieves a file from the specified `file_url` provided in the
    arguments. It uses the "wandb" library for experiment tracking. During the
    process, it creates and uploads an artifact that includes metadata, such as
    the original file URL, to the configured "wandb" project.

    :param args: The input arguments expected to contain the following attributes:
        - file_url (str): The URL of the file to be downloaded.
        - artifact_name (str): The name of the "wandb" artifact to create.
        - artifact_type (str): The type/category of the "wandb" artifact.
        - artifact_description (str): A description of the artifact.

    :returns: None
    """
    basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0]
    logger.info(f"Downloading {args.file_url} ...")
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        logger.info("Creating run")
        with wandb.init(project="random_forest_end_to_end",
                        job_type="download_data") as run:
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()

            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={'original_url': args.file_url}
            )
            artifact.add_file(fp.name, name=basename)

            logger.info("Logging artifact")
            run.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data")
    parser.add_argument("--file_url",
                        type=str,
                        required=True,
                        help="File URL")
    parser.add_argument("--artifact_name",
                        type=str,
                        required=True,
                        help="Artifact name")

    parser.add_argument("--artifact_type",
                        type=str,
                        required=True,
                        help="Artifact type")
    parser.add_argument("--artifact_description",
                        type=str,
                        required=False,
                        help="Artifact description")
    args = parser.parse_args()
    go(args)
