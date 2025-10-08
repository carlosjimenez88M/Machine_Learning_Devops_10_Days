'''
    Chapter #3
    Book: hands on machine learning with scikit-learn and tensorflow
    Download Dataset
    Release Date : 2025-06-11
'''


# =====================#
# ---- Libraries ---- #
# =====================#
import os
import argparse
import pandas as pd
import numpy as np
import wandb
from sklearn.datasets import fetch_openml


import sys
import os


src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from loggers_configuration.logger_config import setup_colored_logger

# ==========================#
#   Logger Configuration   #
# ==========================#

logger = setup_colored_logger("MNIST_Downloader", "INFO")


# =========================#
# ---- Main Function ---- #
# =========================#

def go(args):
    """
    Executes the process of downloading the MNIST dataset, processing it into a DataFrame,
    saving it to a CSV file, logging metrics and dataset metadata to Weights & Biases (W&B),
    and creating a W&B artifact.

    :param args: Command-line arguments required for processing.
        - output_artifact (str): The name of the output CSV file where the dataset will be saved.
        - artifact_name (str): The name assigned to the W&B artifact.
        - artifact_type (str): The type/classification of the W&B artifact.
    :raises Exception: If any error occurs during the dataset fetching, processing,
        or artifact creation process.
    """
    run = wandb.init(project='Classification',
                     job_type="download_data")

    logger.info("🚀 Starting MNIST dataset download from OpenML")

    try:

        logger.info("📥 Fetching MNIST dataset...")
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target

        logger.info("✅ Dataset downloaded successfully")
        logger.info(f"📊 Features shape: {X.shape}")
        logger.info(f"🎯 Target shape: {y.shape}")
        logger.info(f"🔧 Features type: {type(X)}")
        logger.info(f"🏷️  Target type: {type(y)}")


        logger.info("🔄 Converting to pandas DataFrame")


        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]


        df_features = pd.DataFrame(X, columns=feature_names)
        df_target = pd.DataFrame(y, columns=['target'])


        df_mnist = pd.concat([df_features, df_target], axis=1)

        logger.info(f"📋 DataFrame created with shape: {df_mnist.shape}")
        logger.info(f"📈 Target value counts:\n{df_target['target'].value_counts().sort_index()}")


        logger.info("💾 Saving dataset to CSV")
        output_file = args.output_artifact
        df_mnist.to_csv(output_file, index=False)

        logger.info(f"✅ Dataset saved to: {output_file}")


        logger.info("📊 Logging metrics to W&B")
        run.summary["dataset_shape"] = df_mnist.shape
        run.summary["num_features"] = X.shape[1]
        run.summary["num_samples"] = X.shape[0]
        run.summary["num_classes"] = len(np.unique(y))
        run.summary["feature_range"] = [float(X.min()), float(X.max())]


        logger.info("📦 Creating W&B artifact")
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description="MNIST dataset downloaded from OpenML with 784 features (28x28 pixels) and 70k samples"
        )

        artifact.add_file(output_file)


        artifact.metadata = {
            "source": "OpenML mnist_784",
            "num_samples": int(X.shape[0]),
            "num_features": int(X.shape[1]),
            "num_classes": int(len(np.unique(y))),
            "feature_range": [float(X.min()), float(X.max())],
            "target_classes": list(np.unique(y))
        }

        run.log_artifact(artifact)

        logger.info("🎉 Artifact logged successfully")
        logger.info("🏁 MNIST download process completed!")

    except Exception as e:
        logger.error(f"❌ Error downloading MNIST dataset: {e}")
        raise

    finally:
        logger.info("🔄 Finishing W&B run")
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MNIST dataset from OpenML with colored logging",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output CSV file",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the W&B artifact",
        default="mnist_dataset",
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the W&B artifact",
        default="raw_data",
    )

    args = parser.parse_args()

    go(args)