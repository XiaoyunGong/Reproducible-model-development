## load packages
import argparse
import logging.config
import joblib
import yaml
from src.helper import csv_out
from src.preprocess import acquire_data, clean, feature_eng
from src.modeling import my_train_test_split, train, predict, evaluate

# add configuration
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger("run.py")

if __name__ == "__main__":
    # Add the main parser for create and/or add data to the database.
    parser = argparse.ArgumentParser(
        description="Create and/or add data to database")

    subparsers = parser.add_subparsers(dest="subparser_name")

    # Sub-parser to acquire the data
    sp_acquire = subparsers.add_parser("acquire", help="Acquire data")
    sp_acquire.add_argument("--input_path",
                            default=
                            "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data",
                            help="path to the input html")
    sp_acquire.add_argument("--output_path",
                            default=None,
                            help="output path to save the raw data.")

    # Sub-parser to clean the data
    sp_clean = subparsers.add_parser("clean", help="Clean the data")
    sp_clean.add_argument("--config",
                          default="config/data_handling.yaml",
                          help="Path to configuration file")
    sp_clean.add_argument("--input_path_raw",
                          default="data/raw/clouds.data",
                          help="path to pass in the raw data")
    sp_clean.add_argument("--output_path",
                          default=None,
                          help="output path to save the cleaned data.")

    # Sub-parser to feature engineering
    sp_feature_eng = subparsers.add_parser("feature_eng", help="Feature Engineering")
    sp_feature_eng.add_argument("--config",
                                default="config/data_handling.yaml",
                                help="Path to configuration file")
    sp_feature_eng.add_argument("--input_path_clean",
                                default="data/interim/clean.csv",
                                help="path to pass in the clean data")
    sp_feature_eng.add_argument("--target_path",
                                default=None,
                                help="output path to save the target dataset.")
    sp_feature_eng.add_argument("--features_path",
                                default=None,
                                help="output path to save the features dataset.")

    # Sub-parser to train the model
    sp_train = subparsers.add_parser("train", help="training")
    sp_train.add_argument("--config",
                        default="config/modeling.yaml",
                        help="Path to configuration file")
    sp_train.add_argument("--feature_path",
                          default="data/interim/features.csv",
                          help="path to pass in features.")
    sp_train.add_argument("--target_path",
                          default="data/interim/target.csv",
                          help="path to pass in target.")
    sp_train.add_argument("--model_path",
                          default=None,
                          help="output path to save the model.")
    sp_train.add_argument("--X_test_path",
                          default=None,
                          help="output path to save the Xtest.")
    sp_train.add_argument("--y_test_path",
                          default=None,
                          help="output path to save the ytest.")
    sp_train.add_argument("--X_train_path",
                          default=None,
                          help="output path to save the Xtest.")
    sp_train.add_argument("--y_train_path",
                          default=None,
                          help="output path to save the ytest.")

    # Sub-parser to make predictions
    sp_predict = subparsers.add_parser("predict", help="prediction")
    sp_predict.add_argument("--config",
                            default="config/modeling.yaml",
                            help="Path to configuration file")
    sp_predict.add_argument("--model_path",
                            default= "model/random_forest.joblib",
                            help="path to pass in the model.")
    sp_predict.add_argument("--Xtest_path",
                            default="data/interim/X_test.csv",
                            help="path to pass in X_test.")
    sp_predict.add_argument("--ypred_proba_path",
                            default=None,
                            help="output path for the predicted probability.")
    sp_predict.add_argument("--ypred_bin_path",
                            default=None,
                            help="output path for the predicted bin.")

    # Sub-parser to evaluate the model
    sp_eval = subparsers.add_parser("evaluate", help="evaluate")
    sp_eval.add_argument("--ytest_path",
                         default="data/interim/y_test.csv",
                         help="path to pass in ytest")
    sp_eval.add_argument("--ypred_bin_test_path",
                         default="data/pred/class.csv",
                         help="path to pass in ypred_bin_test")
    sp_eval.add_argument("--ypred_proba_test_path",
                         default="data/pred/proba.csv",
                         help="path to pass in ypred_proba_test")
    sp_eval.add_argument("--eval_path",
                         default="output/evaluation.txt",
                         help="path to save the evaluatioin result.")

    args = parser.parse_args()
    sp_used = args.subparser_name

    if sp_used == "acquire":
        acquire_data(input_path=args.input_path, output_path=args.output_path)

    elif sp_used == "clean":
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Configuration file loaded from %s", str(args.config))
        output = clean(**config["clean"], input_path=args.input_path_raw)
        if args.output_path is not None:
            csv_out(output, args.output_path, "cleaned data")

    elif sp_used == "feature_eng":
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Configuration file loaded from %s", str(args.config))

        target, features = feature_eng(input_path=args.input_path_clean, **config["feature_eng"])

        if args.features_path is not None:
            csv_out(features, args.features_path, "features")
        if args.target_path is not None:
            csv_out(target, args.target_path, "target")

    elif sp_used == "train":
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Configuration file loaded from %s", str(args.config))

        X_train, X_test, y_train, y_test = my_train_test_split(**config["my_train_test_split"],
                                                               features_path= args.feature_path,
                                                               target_path=args.target_path)

        if args.X_train_path is not None:
            csv_out(X_train, args.X_train_path, "Xtrain")
        if args.y_train_path is not None:
            csv_out(y_train, args.y_train_path, "ytrain")
        if args.X_test_path is not None:
            csv_out(X_test, args.X_test_path, "Xtest")
        if args.y_test_path is not None:
            csv_out(y_test, args.y_test_path, "ytest")

        rf = train(X_train_path=args.X_train_path, y_train_path=args.y_train_path, **config["train"])

        if args.model_path is not None:
            joblib.dump(rf, args.model_path)
            logger.info("Configuration file loaded from %s", str(args.config))

    elif sp_used == "predict":
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Configuration file loaded from %s", str(args.config))
        ypred_proba_test, ypred_bin_test = predict(**config["predict"],
                                                   model_path=args.model_path,
                                                   Xtest_path=args.Xtest_path)
        if args.ypred_proba_path is not None:
            csv_out(ypred_proba_test, args.ypred_proba_path, "Xtest")
        if args.ypred_bin_path is not None:
            csv_out(ypred_bin_test, args.ypred_bin_path, "ytest")

    elif sp_used == "evaluate":
        evaluate(ytest_path=args.ytest_path, ypred_bin_test_path=args.ypred_bin_test_path,
                 ypred_proba_test_path=args.ypred_proba_test_path, eval_path=args.eval_path)

    else:
        parser.print_help()
