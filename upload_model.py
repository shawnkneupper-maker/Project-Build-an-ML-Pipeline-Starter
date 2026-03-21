import wandb

run = wandb.init(project="nyc_airbnb", job_type="upload_model")

artifact = wandb.Artifact("random_forest_export", type="model")
artifact.add_file("src/train_random_forest/random_forest_dir/model.pkl") # make sure this path is correct

run.log_artifact(artifact, aliases=["prod"])

run.finish()

print("✅ Uploaded as random_forest_export:prod")