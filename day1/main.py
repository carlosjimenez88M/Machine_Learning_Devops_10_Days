'''
    Pipeline Random Forest
    Release Date: 2025-01-26
'''

#=====================#
# ---- libraries ---- #
#=====================#

import mlflow
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

#===============================#
# ---- Hydra Configuration ---- #
#===============================#

@hydra.main(config_path='.', config_name="config", version_base="1.1")

#=========================#
# ---- Main Function ---- #
#=========================#

def go(config: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True
    )
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    if isinstance(config['main']['execute_steps'], str):
        steps_to_execute = config['main']['execute_steps'].split(',')
    else:
        steps_to_execute = list(config['main']['execute_steps'])

    if "1-download_dataset" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "1-download_dataset"),
            entry_point="main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            }
        )
    if "2-clean_dataset" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "2-clean_dataset"),
            "main",
            parameters={
                "input_artifact": "raw_data.csv:latest",
                "artifact_name": "clean_data.csv",
                "artifact_type": "clean_data",
                "artifact_description": "Data with preprocessing applied",
                "n_splits": config["clean"]["n_splits"],
                "test_size": config["clean"]["test_size"],
                "random_state": config["clean"]["random_state"]
            },
        )

        if "3-preprocessing" in steps_to_execute:
            _ = mlflow.run(
                os.path.join(root_path, "3-preprocessing"),
                "main",
                parameters={
                    "input_artifact": "clean_data.csv:latest",
                    "artifact_name": "processing.csv",
                    "artifact_type": "processing",
                    "artifact_description": "Data with preprocessing applied",
                    "output_artifact": "processed_data.csv"
                },
            )
        if "4-segregate" in steps_to_execute:
            _ = mlflow.run(
                os.path.join(root_path, "4-segregate"),
                "main",
                parameters={
                    "input_artifact": "processing.csv:latest",
                    "artifact_root": "data",
                    "artifact_type": "Segregate_Data",
                    "test_size": config["segregate"]["test_size"],
                },
            )
        if "5-model" in steps_to_execute:
            _ = mlflow.run(
                os.path.join(root_path, "5-model"),
                "main",
                parameters={
                    "train_data_artifact": "data_train.csv:latest",
                    "val_size": config["train"]["val_size"],
                    "random_seed": config["train"]["random_seed"]
                },
            )

if __name__ == "__main__":
    go()
