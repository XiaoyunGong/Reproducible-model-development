"""
This module contains helper functions for the feature engineerning part in preprocess.
"""
import logging.config

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def take_log(df: pd.DataFrame,
             old_column: str,
             new_column_name: str) -> pd.DataFrame:
    """this function will take log on one column of the input dataframe.

    Args:
        df (pd.DataFrame): The original df.
        old_column (str): The name of the column that user intended to take log.
        new_column_name (str): The name of the new column (after log)

    Raises:
        KeyError: Raise KeyError if can not find the old_column in the df.

    Returns:
        pd.DataFrame: The df with the column after log.
    """
    all_cols = df.columns.values.tolist()
    check_col_list = [old_column]
    # check if all variables in the input is in the df
    if all(elem in all_cols for elem in check_col_list):
        df[new_column_name] = df[old_column].apply(np.log)
    else:
        logger.error("Can not find old_column in the dataset (function:take_log).")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")
    return df

def multiplication(df: pd.DataFrame,
                   old_column1: str,
                   old_column2: str,
                   new_column_name: str) -> pd.DataFrame:
    """This function will take multiplication of two columns in the input df.

    Args:
        df (pd.DataFrame): The original df.
        old_column1 (str): The name of the first column that user intended to take multiplication.
        old_column2 (str): The name of the second column that user intended to take multiplication.
        new_column_name (str): The name of the multiplied column.

    Raises:
        KeyError: Raise KeyError if can not find the old_column(s) in the df.

    Returns:
        pd.DataFrame: The df with the column after multiplication.
    """
    all_cols = df.columns.values.tolist()
    check_col_list = [old_column1, old_column2]
    # check if all variables in the input is in the df
    if all(elem in all_cols for elem in check_col_list):
        df[new_column_name] = df[old_column1].multiply(df[old_column2])
    else:
        logger.error("Can not find some old columns in the dataset (function:multiplication).")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")
    return df

def subtraction(df: pd.DataFrame,
             old_column1: str,
             old_column2: str,
             new_column_name: str) -> pd.DataFrame:
    """This function will take subtraction of two columns in the input df.

    Args:
        df (pd.DataFrame): The original df.
        old_column1 (str): The name of the minuend column.
        old_column2 (str): The name of the subtrahend column
        new_column_name (str): The name of the subtraction result column.
    Raises:
        KeyError: Raise KeyError if can not find the old_column(s) in the df.

    Returns:
        pd.DataFrame: The df with the column after subtraction.
    """
    all_cols = df.columns.values.tolist()
    check_col_list = [old_column1, old_column2]
    # check if all variables in the input is in the df
    if all(elem in all_cols for elem in check_col_list):
        df[new_column_name] = df[old_column1] - df[old_column2]
    else:
        logger.error("Can not find some old columns in the dataset (function:subtraction).")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")
    return df

def norm_range(df: pd.DataFrame,
              old_column1: str,
              old_column2: str,
              old_column3: str,
              new_column_name: str) -> pd.DataFrame:
    """This function will normalize the difference of two columns using a third column.

    Args:
        df (pd.DataFrame): The original df
        old_column1 (str): The name of the minuend column
        old_column2 (str): The name of the subtrahend column
        old_column3 (str): The name of the third column (divisor)
        new_column_name (str): The name of the resulting column.

    Raises:
        KeyError: Raise KeyError if can not find the old_column(s) in the df.

    Returns:
        pd.DataFrame:  The df with the column after normalization.
    """
    all_cols = df.columns.values.tolist()
    check_col_list = [old_column1, old_column2, old_column3]
    # check if all variables in the input is in the df
    if all(elem in all_cols for elem in check_col_list):
        df[new_column_name] = (df[old_column1] - df[old_column2]).divide(df[old_column3])
    else:
        logger.error("Can not find some old columns in the dataset (function:norm_range).")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")
    return df
