#!/usr/bin/env python
"""
This script trains a Random Forest
"""

import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import json
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

import wandb


def delta_date_feature(dates):
    """
    Convert date into number of days from most recent review
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Load RF configuration
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)

    run.config.update(rf_config)

    rf_config["random_state"] = args.random_seed

    # Download dataset artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=X[args.stratify_by],
        random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(
        rf_config,
        args.max_tfidf_features
    )

    # ---------------------------------------
    # FIT MODEL
    # ---------------------------------------
    logger.info("Fitting model")

    sk_pipe.fit(X_train, y_train)

    # ---------------------------------------
    # EVALUATION
    # ---------------------------------------
    logger.info("Scoring")

    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"R2: {r_squared}")
    logger.info(f"MAE: {mae}")

    # ---------------------------------------
    # SAVE MODEL
    # ---------------------------------------
    logger.info("Exporting model")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        input_example=X_train.iloc[:5]
    )

    # Upload model artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained random forest model",
        metadata=rf_config,
    )

    artifact.add_dir("random_forest_dir")

    run.log_artifact(artifact)

    # ---------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    run.log({
        "feature_importance": wandb.Image(fig_feat_imp)
    })


def plot_feature_importance(pipe, feat_names):

    feat_imp = pipe["random_forest"].feature_importances_[:len(feat_names)-1]

    nlp_importance = sum(
        pipe["random_forest"].feature_importances_[len(feat_names)-1:]
    )

    feat_imp = np.append(feat_imp, nlp_importance)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.bar(range(feat_imp.shape[0]), feat_imp)

    ax.set_xticks(range(feat_imp.shape[0]))
    ax.set_xticklabels(np.array(feat_names), rotation=90)

    fig.tight_layout()

    return fig


def get_inference_pipeline(rf_config, max_tfidf_features):

    ordinal_categorical = ["room_type"]

    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder()

    # ---------------------------------------
    # NON ORDINAL CATEGORICAL
    # ---------------------------------------
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # ---------------------------------------
    # NUMERIC FEATURES
    # ---------------------------------------
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]

    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # ---------------------------------------
    # DATE FEATURE
    # ---------------------------------------
    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, validate=False)
    )

    # ---------------------------------------
    # NLP FEATURE
    # ---------------------------------------
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})

    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english"
        )
    )

    # ---------------------------------------
    # COLUMN TRANSFORMER
    # ---------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("zero_impute", zero_imputer, zero_imputed),
            ("date", date_imputer, ["last_review"]),
            ("name", name_tfidf, ["name"])
        ],
        remainder="drop"
    )

    processed_features = (
        ordinal_categorical
        + non_ordinal_categorical
        + zero_imputed
        + ["last_review", "name"]
    )

    random_forest = RandomForestRegressor(**rf_config)

    # ---------------------------------------
    # INFERENCE PIPELINE
    # ---------------------------------------
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--trainval_artifact", type=str)

    parser.add_argument("--val_size", type=float)

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--stratify_by", type=str)

    parser.add_argument("--rf_config")

    parser.add_argument("--max_tfidf_features", type=int, default=10)

    parser.add_argument("--output_artifact", type=str)

    args = parser.parse_args()

    go(args)
