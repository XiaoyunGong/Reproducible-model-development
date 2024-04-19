# MSiA423 Assignment 3 : Reproducible model development

Author: Xiaoyun Gong

## Step-by-step model development

### Build the docker image
The folllowing command will build the docker image for producing each step's artifact.

```bash
docker build -f dockerfiles/Dockerfile -t clouds .
```
or
```bash
make image
```

### Data acquisition 
To acqurire the raw daya and save it to the `data/raw/clouds.data`, run

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py acquire --output_path=data/raw/clouds.data 
```
or
```bash
make acquire
```

### Data cleaning
To create the cleaned dataset and save it to the `data/interim/clean.csv`, run
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py clean --output_path=data/interim/clean.csv
```
or
```bash
make clean
```

### Feature engineering
To generate the features and save them to the `data/interim/target.csv` and `data/interim/features.csv`, run
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py feature_eng  --features_path=data/interim/features.csv --target_path=data/interim/target.csv 
```
or
```bash
make feature_eng
```

### Model training
To generate the trained model object and save it to the `model/random_forest.joblib` and run
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py train --model_path=model/random_forest.joblib --X_test_path=data/interim/X_test.csv --y_test_path=data/interim/y_test.csv --X_train_path=data/interim/X_train.csv --y_train_path=data/interim/y_train.csv 
```
or
```bash
make train
```
This function will also save X_train, X_test, y_train, y_test to the interim folder.

### Prediction
To produce predictions for evaluating the model and save predicted proba to `data/pred/proba.csv` and predicted class to `data/pred/class.csv`, run
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py predict --ypred_proba_path=data/pred/proba.csv --ypred_bin_path=data/pred/class.csv
```
or
```bash
make predict
```

### Model evaluation
To compute the performance metrics, print them out, and save it as a .txt file to `output/evaluation.txt`, run
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py evaluate
```
or
```bash
make evaluate
```

## Pipeline model deployment

### Build the docker image
The folllowing command will build the docker image for producing the entire model pipeline.

```bash
docker build -f dockerfiles/Dockerfile -t clouds .
```

### Run the pipeline
```bash
bash pipeline.sh
```
or 
```bash
make all
```

## Testing

These commands will build a testing image and run unit tests
```bash
docker build -f dockerfiles/Dockerfile.test -t cloud-tests .
docker run cloud-tests
```