import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_name="config")
def go(config):

    # Steps to execute
    steps_to_execute = config["main"]["steps"].split(",")

    # repository containing components
    components_repository = config["main"]["components_repository"]

    # Random forest config file
    rf_config = os.path.abspath(config["modeling"]["random_forest_config"])

    # -------------------------------------------------------
    # DOWNLOAD STEP
    # -------------------------------------------------------
    if "download" in steps_to_execute:

        _ = mlflow.run(
            f"{components_repository}/download",
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            },
        )

    # -------------------------------------------------------
    # BASIC CLEANING STEP
    # -------------------------------------------------------
    if "basic_cleaning" in steps_to_execute:

        _ = mlflow.run(
            f"{components_repository}/basic_cleaning",
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": config["data"]["cleaned_artifact"],
                "output_type": config["data"]["cleaned_artifact_type"],
                "output_description": config["data"]["cleaned_artifact_description"],
            },
        )

    # -------------------------------------------------------
    # DATA CHECK STEP
    # -------------------------------------------------------
    if "data_check" in steps_to_execute:

        _ = mlflow.run(
            f"{components_repository}/data_check",
            "main",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": config["data"]["reference_dataset"],
                "kl_threshold": config["data"]["kl_threshold"],
            },
        )

    # -------------------------------------------------------
    # DATA SPLIT STEP
    # -------------------------------------------------------
    if "data_split" in steps_to_execute:

        _ = mlflow.run(
            f"{components_repository}/train_val_test_split",
            "main",
            parameters={
                "input_artifact": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
        )

    # -------------------------------------------------------
    # TRAIN RANDOM FOREST STEP
    # -------------------------------------------------------
    if "train_random_forest" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(
                hydra.utils.get_original_cwd(),
                "src",
                "train_random_forest",
            ),
            "main",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "output_artifact": "random_forest_export",
                "rf_config": rf_config,
                "random_seed": config["modeling"]["random_seed"],
                "val_size": config["modeling"]["val_size"],
                "stratify_by": config["modeling"]["stratify_by"],
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
            },
        )


if __name__ == "__main__":
    go()
