"""
This is the unit testing file for the feature_eng function in the preprocess module.
"""

## import packages
import logging.config
import pytest

import pandas as pd

import src.features_helper

logger = logging.getLogger(__name__)

def test_take_log():
    """This is the testing function for the take_log function. (happy path)
    """
    df_in_log = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3"])

    df_true = pd.DataFrame(
        [[10,20,30, 2.30258],
         [1, 2, 3, 0],
         [100, 200, 300, 4.60517]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3", "log_col1"])

    logger.info("running take_log happy test.")

    # Compute test output
    df_test = src.features_helper.take_log(df=df_in_log, old_column="col1", new_column_name="log_col1")

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_take_log_missing_col():
    """This is the testing function for the take_log function. (unhappy path)
    """
    df_false = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1_missing", "col2", "col3"])
    logger.info("running take_log unhappy test.")
    with pytest.raises(KeyError):
        src.features_helper.take_log(df=df_false, old_column="col1", new_column_name="log_col1")

def test_multiplication():
    """This is the testing function for the multiplication function. (happy path)
    """
    df_in_mul = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3"])

    df_true = pd.DataFrame(
        [[10,20,30, 200],
         [1, 2, 3, 2],
         [100, 200, 300,20000]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3", "col2xcol1"])
    logger.info("running multiplication happy test.")
    # Compute test output
    df_test = src.features_helper.multiplication(df=df_in_mul, old_column1="col1",
                                                 old_column2="col2", new_column_name="col2xcol1")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_multiplication_missing_col():
    """This is the testing function for the multiplication function. (unhappy path)
    """
    df_unhappy = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1_missing", "col2", "col3"])
    logger.info("running multiplication unhappy test.")
    with pytest.raises(KeyError):
        src.features_helper.multiplication(df=df_unhappy, old_column1="col1",
                                           old_column2="col2", new_column_name="col2xcol1")

def test_subtraction():
    """This is the testing function for the subtraction function. (happy path)
    """
    df_in_sub = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3"])

    df_true = pd.DataFrame(
        [[10,20,30, 10],
         [1, 2, 3, 1],
         [100, 200, 300,100]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3", "col2-col1"])
    logger.info("running subtraction happy test.")
    # Compute test output
    df_test = src.features_helper.subtraction(df=df_in_sub, old_column1="col2",
                                                 old_column2="col1", new_column_name="col2-col1")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_subtraction_missing_col():
    """This is the testing function for the subtraction function. (unhappy path)
    """
    df_unhappy = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1_missing", "col2", "col3"])
    logger.info("running subtraction unhappy test.")
    with pytest.raises(KeyError):
        src.features_helper.multiplication(df=df_unhappy, old_column1="col1",
                                           old_column2="col2", new_column_name="col2-col1")

def test_norm_range():
    """This is the testing function for the norm_range function. (happy path)
    """
    df_in_norm = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3"])

    df_true = pd.DataFrame(
        [[10,20,30, 1.0],
         [1, 2, 3, 1.0],
         [100, 200, 300,1.0]],
        index=[2, 3, 5],
        columns=["col1", "col2", "col3", "(col3-col2)/col1"])
    logger.info("running norm_range happy test.")
    # Compute test output
    df_test = src.features_helper.norm_range(df=df_in_norm, old_column1="col3",
                                                 old_column2="col2", old_column3 = "col1",
                                                 new_column_name="(col3-col2)/col1")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_norm_range_missing_col():
    """This is the testing function for the norm_range function. (unhappy path)
    """
    df_unhappy = pd.DataFrame(
        [[10,20,30],
         [1, 2, 3],
         [100, 200, 300]],
        index=[2, 3, 5],
        columns=["col1_missing", "col2", "col3"])
    logger.info("running norm_range unhappy test.")
    with pytest.raises(KeyError):
        src.features_helper.norm_range(df=df_unhappy, old_column1="col3",
                                       old_column2="col2", old_column3 = "col1",
                                       new_column_name="(col3-col2)/col1")
