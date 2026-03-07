#!/usr/bin/env python
"""
Download a file and upload it as W&B artifact
"""
import argparse
import logging
import os
import wandb
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def go(args):
    run = wandb.init(job_type="download_file")
    run.config.update(args)

    local_path = os.path.join("data", args.sample)
    logger.info(f"Uploading {args.artifact_name} to W&B from {local_path}")

    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        local_path,
        run
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)
    args = parser.parse_args()
    go(args)
