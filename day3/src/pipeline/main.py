'''
    Pipeline Post Covid Model
    Release Date: 2025-09-07
'''


import os
import shutil
import glob

import hydra
import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf


def cleanup_temp_files(root_path):
    """Clean up temporary MLflow and Wandb files"""
    try:
        mlruns_dirs = glob.glob(os.path.join(root_path, "**/mlruns"), recursive=True)
        for mlruns_dir in mlruns_dirs:
            if os.path.exists(mlruns_dir):
                shutil.rmtree(mlruns_dir)
                print(f"Cleaned up: {mlruns_dir}")

        hydra_dirs = glob.glob(os.path.join(root_path, "**/.hydra"), recursive=True)
        for hydra_dir in hydra_dirs:
            if os.path.exists(hydra_dir):
                shutil.rmtree(hydra_dir)
                print(f"Cleaned up: {hydra_dir}")

        artifacts_dirs = glob.glob(os.path.join(root_path, "**/artifacts"), recursive=True)
        for artifacts_dir in artifacts_dirs:
            if os.path.exists(artifacts_dir):
                shutil.rmtree(artifacts_dir)
                print(f"Cleaned up: {artifacts_dir}")

        wandb_dirs = glob.glob(os.path.join(root_path, "**/wandb"), recursive=True)
        for wandb_dir in wandb_dirs:
            if os.path.exists(wandb_dir):
                shutil.rmtree(wandb_dir)
                print(f"Cleaned up: {wandb_dir}")

        outputs_dirs = glob.glob(os.path.join(root_path, "**/outputs"), recursive=True)
        for outputs_dir in outputs_dirs:
            if os.path.exists(outputs_dir):
                shutil.rmtree(outputs_dir)
                print(f"Cleaned up: {outputs_dir}")

    except Exception as e:
        print(f"Warning: Could not clean up some temporary files: {e}")


@hydra.main(config_path='.', config_name="config", version_base="1.1")
def go(config: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    if isinstance(config['main']['execute_steps'], str):
        steps_to_execute = config['main']['execute_steps'].split(',')
    else:
        steps_to_execute = list(config['main']['execute_steps'])

    if "1-load_dataset" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "1-load_dataset"),
            entry_point="main",
            parameters={
                "file": config["data"]["file"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
                "kaggle_dataset": config["data"]["kaggle_dataset"]
            }
        )

    if "2-preprocessing_dataset" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "2-preprocessing_dataset"),
            entry_point="main",
            parameters={
                "input_artifact": config["preprocessing"]["input_artifact"],
                "output_artifact": config["preprocessing"]["output_artifact"],
                "artifact_name": config["preprocessing"]["output_artifact_name"],
                "artifact_type": config["preprocessing"]["output_artifact_type"],
                "artifact_description": config["preprocessing"]["output_artifact_description"]
            }
        )
    
    if "3-split_dataset" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "3-split_dataset"),
            entry_point="main",
            parameters={
                "input_artifact": config["split"]["input_artifact"],
                "train_artifact": config["split"]["train_artifact_name"],
                "test_artifact": config["split"]["test_artifact_name"],
                "artifact_type": config["split"]["artifact_type"],
                "artifact_description": config["split"]["artifact_description"]
            }
        )

    if "4-training_model" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "4-training_model"),
            entry_point="main",
            parameters={
                "input_artifact": config["training"]["input_artifact"],
                "output_artifact": config["training"]["output_artifact"],
                "output_type": config["training"]["output_type"],
                "output_description": config["training"]["output_description"],
                "n_estimators": config["training"]["n_estimators"],
                "max_depth": config["training"]["max_depth"],
                "min_samples_split": config["training"]["min_samples_split"],
                "min_samples_leaf": config["training"]["min_samples_leaf"],
                "max_features": config["training"]["max_features"],
                "bootstrap": config["training"]["bootstrap"],
                "random_state": config["training"]["random_state"]
            }
        )


    if "5-evaluate_model" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "5-evaluate_model"),
            entry_point="main",
            parameters={
                "test_data": config["evaluate"]["test_data"],
                "model_export": config["evaluate"]["model_export"]
            }
        )



if __name__ == "__main__":
    go()
    
