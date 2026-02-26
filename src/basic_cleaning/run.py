import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):

    # Initialize W&B run
    run = wandb.init(project="nyc_airbnb", group="basic_cleaning", save_code=True)

    logger.info(f"Downloading artifact {args.input_artifact}")

    # Download input artifact
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Load dataset
    df = pd.read_csv(artifact_path)
    logger.info(f"Loaded dataset with {len(df)} rows")

    # Filter price range
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save cleaned file
    clean_path = "clean_sample.csv"
    df.to_csv(clean_path, index=False)
    logger.info("Saved cleaned dataset")

    # Create output artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(clean_path)
    run.log_artifact(artifact)

    run.finish()
    logger.info("Cleaning step completed successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Basic cleaning step that filters price and converts dates"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Name of the input artifact to download"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of the cleaned dataset artifact"
    )

    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type of the output artifact"
    )

    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description of the cleaned dataset artifact"
    )

    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum price threshold"
    )

    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum price threshold"
    )

    args = parser.parse_args()

    go(args)
