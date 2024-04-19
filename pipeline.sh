#!/usr/bin/env bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py acquire --output_path=data/raw/clouds.data

docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py clean --output_path=data/interim/clean.csv

docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py feature_eng  --features_path=data/interim/features.csv --target_path=data/interim/target.csv

docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py train --model_path=model/random_forest.joblib --X_test_path=data/interim/X_test.csv --y_test_path=data/interim/y_test.csv --X_train_path=data/interim/X_train.csv --y_train_path=data/interim/y_train.csv

docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py predict --ypred_proba_path=data/pred/proba.csv --ypred_bin_path=data/pred/class.csv

docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py evaluate