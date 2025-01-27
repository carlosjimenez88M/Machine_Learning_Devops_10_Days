'''
            Step #1 Download Dataset
    Description: Program for generating the download
    of databases for the pipeline with Weights and Biases.
    Release Date: 2025-01-26
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import argparse
import logging
import pathlib
import wandb
import requests
import tempfile

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#

def go(args):
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