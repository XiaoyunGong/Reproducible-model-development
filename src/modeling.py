"""
This module include functions that is related to modeling. (Train test split, train, and predict.)
"""
import logging.config
from typing import List, Tuple
import joblib
import sklearn
import sklearn.ensemble
import pandas as pd

from src.helper import csv_in

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name

def my_train_test_split(test_size: float,
                        features_path: str,
                        target_path: str,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """perform train test split on a dataset.

    Args:
        test_size (float): test size for the split.
        features_path (str): The path that store the input .csv file for features.
        target_path (str): The path that store the input .csv fiel for the targets.
        random_state (int): The random state for reproducibility purpose (default to 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
    """
    # validate and read in csv file
    if not isinstance(features_path, str):
        logger.error("The input features_path is not a string")
        raise TypeError("Input features_path expecting a string as a value, not %s." %str({type(features_path)}))

    if not isinstance(target_path, str):
        logger.error("The input target_path is not a string")
        raise TypeError("Input target_path expecting a string as a value, not %s." % str({type(target_path)}))

    features = csv_in(features_path)
    target = csv_in(target_path)

    # validate other inputs
    if not isinstance(test_size, float):
        logger.error("The input test_size is not a float")
        raise TypeError("Input test_size expecting a float as a value, not %s." % str({type(test_size)}))

    if not isinstance(random_state, int):
        logger.error("The input random_state is not an integer.")
        raise TypeError("Input random_state expecting a string as a value, not %s."% str({type(random_state)}))

    #train test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, target.values.ravel(), test_size=test_size, random_state = random_state)

    logger.info("Performed train test split with test size = %f",  test_size)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    logger.debug("The data type of y_test is %s", str(type(y_test)))

    return X_train, X_test, y_train, y_test

def train(X_train_path: str,
          y_train_path: str,
          initial_features: List,
          n_estimators: int,
          max_depth: int,
          random_state: int
          ) -> sklearn.base.BaseEstimator:
    """This function will train the model using random forest.


    Args:
        X_train_path (str): The path that stored the X_train .csv file
        y_train_path (str): The path that stored the y_train .csv file
        initial_features (List): The list of initial features used in the model
        n_estimators (int): The number of estimators for the random forest model
        max_depth (int): The max_depth of the random forest model
        random_state (int): The random state of the random forest model

    Returns:
        sklearn.base.BaseEstimator: random forest model
    """

    # validate other inputs

    if not isinstance(n_estimators, int):
        logger.error("The input n_estimators is not an integer.")
        raise TypeError("Input n_estimators expecting a string as a value, not %s."% str({type(n_estimators)}))
    if not isinstance(max_depth, int):
        logger.error("The input max_depth is not an integer.")
        raise TypeError("Input n_estimators expecting a string as a value, not %s."% str({type(max_depth)}))
    if not isinstance(random_state, int):
        logger.error("The input random_state is not an integer.")
        raise TypeError("Input random_state expecting a string as a value, not %s."% str({type(random_state)}))

    X_train = csv_in(X_train_path)
    y_train = csv_in(y_train_path).values.ravel()

    logger.debug("y_train is a %s and has shape %s", str(type(y_train)), str(y_train.shape))

    # check if all elements in the initial features list exist in the features df
    all_cols = X_train.columns.values.tolist()
    if all(elem in all_cols for elem in initial_features):
        pass
    else:
        logger.error("Some columns in the columns input is not in the dataset. Check again!")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")

    #train the model
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                 max_depth=max_depth,
                                                 random_state=random_state)
    rf.fit(X_train[initial_features], y_train)

    # save the model and df's
    return rf

def predict(model_path: str,
            Xtest_path: str,
            initial_features: List
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This function will make the prediction (proba and class) using the model and X_test fed in.

    Args:
        model_path (str): The path of the model joblib file
        Xtest_path (str): The path of the X_test csv file
        initial_features (List): The list of features used

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The predicted probablity dataframe and the predicted bin dataframe.
    """

    # load the model
    if not isinstance(model_path, str):
        logger.error("The input model_path is not a String.")
        raise TypeError("Input model_path expecting a string as a value, not %s."% str({type(model_path)}))
    rf = joblib.load(model_path)
    # load the dataframe
    if not isinstance(Xtest_path, str):
        logger.error("The input model_path is not a String.")
        raise TypeError("Input model_path expecting a string as a value, not %s."% str({type(model_path)}))
    X_test = csv_in(Xtest_path)

    # make predictions
    ypred_proba_test = pd.DataFrame(rf.predict_proba(X_test[initial_features])[:,1])
    ypred_bin_test = pd.DataFrame(rf.predict(X_test[initial_features]))

    return ypred_proba_test, ypred_bin_test

def evaluate(ytest_path: str,
            ypred_proba_test_path: str,
            ypred_bin_test_path: str,
            eval_path: str) -> None:
    """This function will print out the evaluation on test set,
       and generate a .txt file for the info printed.

    Args:
        ytest_path (str): The path of y_test csv file.
        ypred_proba_test_path (str): The path of ypred_proba_test csv file.
        ypred_bin_test_path (str): The path of ypred_bin_test csv file.
        eval_path (str): The path to save the evaluation result txt file.
    """
    # readin the csv files as df
    if not isinstance(ypred_proba_test_path, str):
        logger.error("The input ypred_proba_test_path is not a String.")
        raise TypeError("Input ypred_proba_test_path expecting a string as a value, not %s."
                        % str({type(ypred_proba_test_path)}))
    if not isinstance(ypred_bin_test_path, str):
        logger.error("The input ypred_bin_test_path is not a String.")
        raise TypeError("Input ypred_bin_test_path expecting a string as a value, not %s."
                        % str({type(ypred_bin_test_path)}))
    if not isinstance(ytest_path, str):
        logger.error("The input ytest_path is not a String.")
        raise TypeError("Input ytest_path expecting a string as a value, not %s." % str({type(ytest_path)}))

    ypred_proba_test_df = csv_in(ypred_proba_test_path)
    ypred_bin_test_df = csv_in(ypred_bin_test_path)
    y_test = csv_in(ytest_path)

    # check other input
    if not isinstance(eval_path, str):
        logger.error("The input eval_path is not a String.")
        raise TypeError("Input eval_path expecting a string as a value, not %s." % str({type(eval_path)}))
    # convert to Numpy
    ypred_proba_test = ypred_proba_test_df.to_numpy()
    ypred_bin_test = ypred_bin_test_df.to_numpy()
    y_test = y_test.to_numpy()

    # evaluation
    auc = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)
    confusion = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)

    print("AUC on test: %0.3f" % auc)
    print("Accuracy on test: %0.3f" % accuracy)
    print()
    eval_df = pd.DataFrame(confusion,
                    index=["Actual negative","Actual positive"],
                    columns=["Predicted negative", "Predicted positive"])
    print(eval_df)

    eval_str = eval_df.to_string()

    # write to a output file
    with open(eval_path, "w") as out:
        out.write("AUC on test: %0.3f" % auc + "\n")
        out.write("Accuracy on test: %0.3f" % accuracy + "\n")
        out.write(eval_str)
