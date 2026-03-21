#!/usr/bin/env python
"""
Train a Random Forest model with W&B + MLflow alignment
"""

import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def delta_date_feature(dates):
    """Convert date into number of days from most recent review"""
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


def get_inference_pipeline(rf_config, max_tfidf_features):
    # Column preprocessing
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]

    # Pipelines
    ordinal_preproc = OrdinalEncoder()
    non_ordinal_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, validate=False)
    )
    reshape_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_1d,
        TfidfVectorizer(max_features=max_tfidf_features, stop_words="english")
    )

    preprocessor = ColumnTransformer([
        ("ordinal_cat", ordinal_preproc, ordinal_categorical),
        ("non_ordinal_cat", non_ordinal_preproc, non_ordinal_categorical),
        ("zero_impute", zero_imputer, zero_imputed),
        ("date", date_imputer, ["last_review"]),
        ("name", name_tfidf, ["name"])
    ], remainder="drop")

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    rf = RandomForestRegressor(**rf_config)
    sk_pipe = Pipeline([("preprocessor", preprocessor), ("random_forest", rf)])
    return sk_pipe, processed_features


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[:len(feat_names)-1]
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names)-1:])
    feat_imp = np.append(feat_imp, nlp_importance)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(range(feat_imp.shape[0]), feat_imp)
    ax.set_xticks(range(feat_imp.shape[0]))
    ax.set_xticklabels(np.array(feat_names), rotation=90)
    fig.tight_layout()
    return fig


def go(args):
    # Start W&B run
    run = wandb.init(job_type="train_random_forest")
    run.config.update(vars(args))

    # Parse RF config
    import ast
    rf_config = ast.literal_eval(args.rf_config)
    rf_config["random_state"] = args.random_seed
    run.config.update(rf_config)

    # Load dataset artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")
    logger.info(f"Price range: min={y.min()} max={y.max()}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    # Pipeline
    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    # Train model
    sk_pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = sk_pipe.predict(X_val)
    r2 = sk_pipe.score(X_val, y_val)
    mae = mean_absolute_error(y_val, y_pred)
    logger.info(f"R2: {r2}, MAE: {mae}")

    # Save locally
    model_path = "model"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    mlflow.sklearn.save_model(
        sk_pipe,
        model_path,
        input_example=X_train.iloc[:5]
    )

    # Log artifact to W&B (type="model")
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model",
        description="Trained random forest model",
        metadata=rf_config
    )
    artifact.add_dir(model_path)
    run.log_artifact(artifact)

    # Feature importance plot
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    run.summary["r2"] = r2
    run.summary["mae"] = mae
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, default=10)
    parser.add_argument("--output_artifact", type=str, default="random_forest_export")
    args = parser.parse_args()
    go(args)
