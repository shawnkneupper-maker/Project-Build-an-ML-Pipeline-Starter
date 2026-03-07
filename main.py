#!/usr/bin/env python
"""
Main entry point for the ML pipeline
"""
import mlflow
import yaml
import os

if __name__ == "__main__":
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Determine which steps to run
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_name="config", config_path=None)
    def run_pipeline(cfg: DictConfig):
        steps = cfg.main.steps

        if "download" in steps:
            mlflow.run(
                f"{cfg.main.components_repository}/get_data",
                entry_point="main",
                parameters={
                    "sample": cfg.data.raw_sample,
                    "artifact_name": cfg.data.raw_artifact_name,
                    "artifact_type": cfg.data.raw_artifact_type,
                    "artifact_description": cfg.data.raw_artifact_description
                },
            )

        if "basic_cleaning" in steps:
            mlflow.run(
                f"{cfg.main.components_repository}/basic_cleaning",
                entry_point="main",
                parameters={
                    "input_artifact": f"{cfg.data.raw_artifact_name}:latest",
                    "output_artifact": cfg.data.clean_artifact_name
                },
            )

        if "data_check" in steps:
            mlflow.run(
                f"{cfg.main.components_repository}/data_check",
                entry_point="main",
                parameters={
                    "input_artifact": f"{cfg.data.clean_artifact_name}:latest",
                    "min_price": cfg.modeling.min_price,
                    "max_price": cfg.modeling.max_price
                },
            )

        if "train_val_test_split" in steps:
            mlflow.run(
                f"{cfg.main.components_repository}/train_val_test_split",
                entry_point="main",
                parameters={
                    "input_artifact": f"{cfg.data.clean_artifact_name}:latest",
                    "test_size": cfg.modeling.test_size,
                    "random_seed": cfg.modeling.random_seed,
                    "stratify_by": cfg.modeling.stratify_by
                },
            )

        if "train_random_forest" in steps:
            mlflow.run(
                f"{cfg.main.components_repository}/train_random_forest",
                entry_point="main",
                parameters={
                    "trainval_artifact": f"{cfg.data.trainval_artifact}:latest",
                    "val_size": cfg.modeling.val_size,
                    "random_seed": cfg.modeling.random_seed,
                    "stratify_by": cfg.modeling.stratify_by,
                    "rf_config": cfg.modeling.random_forest_config,
                    "max_tfidf_features": cfg.modeling.max_tfidf_features,
                    "output_artifact": cfg.modeling.random_forest_artifact
                },
            )

    run_pipeline()
