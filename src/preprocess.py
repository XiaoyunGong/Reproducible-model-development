import logging.config
import subprocess
import sys
from typing import List, Tuple

import pandas as pd
import numpy as np
from requests import HTTPError
from src.features_helper import multiplication, norm_range, subtraction, take_log
from src.helper import csv_in

logger = logging.getLogger(__name__)

def acquire_data(output_path: str, input_path:str) -> None:
    """This command will acquire data from the UCI database.

    Args:
        output_path (str): the path that save the output file.
        input_path (str): the path that the data is pulled from.

    """
    # check the input type
    if not isinstance(input_path, str):
        logger.error("The input input_path is not a string.")
        raise TypeError("Input input_path expecting a string as a value, not %s." % str({type(input_path)}))

    if not isinstance(output_path, str):
        logger.error("The input output_path is not a string.")
        raise TypeError("Input output_path expecting a string as a value, not %s." % str({type(output_path)}))

    # use subprocess to read in the data
    logger.info("trying to acquire the data from %s", input_path)
    try:
        subprocess.run(["curl", "-o", output_path, input_path], capture_output=True, check=True)
        logger.info("data acquired and saved to %s", output_path)
    except HTTPError:
        logger.error("Page does not exist")
        sys.exit(1)
    except Exception as err:
        logger.error("Unexpected error: %s", err)
        raise err

def clean(input_path: str,
          columns: List,
          first_cloud_start: int,
          first_cloud_end: int,
          second_cloud_start: int,
          second_cloud_end: int) -> pd.DataFrame:
    """This function will clean the data.
       First, this function will parse through the .data file read in.
       Then, it will collect info on first type of cloud and second type of cloud.
       Afterwards, it will merge these two dataframes together.

    Args:
        input_path (str): the location of .data file.
        columns (List): useful columns.
        first_cloud_start (int): # of rows where the info on first cloud starts.
        first_cloud_end (int): # of rows where the info on first cloud ends.
        second_cloud_start (int): # of rows where the info on second cloud starts.
        second_cloud_end (int): # of rows where the info on second cloud ends.
        output_path (str): output path of the .csv file.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """

    # check input type
    if not isinstance(input_path, str):
        logger.error("The input 'input_path' is not a string.")
        raise TypeError("Input 'input_path' expecting a string as a value, not %s."%str({type(input_path)}))

    if not isinstance(first_cloud_start, int):
        logger.error("The input first_cloud_start is not an integer.")
        raise TypeError("Input first_cloud_start expecting a string as a value, not %s."
                        % str({type(first_cloud_start)}))

    if not isinstance(first_cloud_end, int):
        logger.error("The input first_cloud_end is not an integer.")
        raise TypeError("Input first_cloud_end expecting a string as a value, not %s." % str({type(first_cloud_end)}))

    if not isinstance(second_cloud_start, int):
        logger.error("The input second_cloud_start is not an integer.")
        raise TypeError("Input second_cloud_start expecting a string as a value, not %s."
                        % str({type(second_cloud_start)}))

    if not isinstance(second_cloud_end, int):
        logger.error("The input second_cloud_end is not an integer.")
        raise TypeError("Input second_cloud_end expecting a string as a value, not %s." % str({type(second_cloud_end)}))

    # check if the bounds are not out of bound
    if first_cloud_start < 53 or \
        first_cloud_end >1077 or \
        second_cloud_end <1082 or \
        second_cloud_end >2105:
        logger.warning("potential index out of bound. Recommend to check the boundary of first/second clouds.")

    # read in data
    try:
        with open(input_path,"r") as file:
            data = [[s for s in line.split(" ") if s!=""] for line in file.readlines()]
            logger.info("Data read in from %s", input_path)
    except FileNotFoundError:
        logger.error("Can't read from the input location %s Check again!", input_path)
        sys.exit(1)

    # handle the first cloud df
    first_cloud = data[first_cloud_start:first_cloud_end]

    first_cloud = [[float(s.replace("/n", "")) for s in cloud]
               for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)

    first_cloud["class"] = np.zeros(len(first_cloud))
    logger.debug("The %i to %i records were saved as first_cloud.", first_cloud_start, first_cloud_end)

    # handle the second cloud df
    second_cloud = data[second_cloud_start:second_cloud_end]

    second_cloud = [[float(s.replace("/n", "")) for s in cloud]
                for cloud in second_cloud]

    second_cloud = pd.DataFrame(second_cloud, columns=columns)

    second_cloud["class"] = np.ones(len(second_cloud))
    logger.debug("The %i to %i records were saved as second_cloud.", second_cloud_start, second_cloud_end)

    data = pd.concat([first_cloud, second_cloud])

    return data

def feature_eng(input_path: str,
                columns: List,
                log_column: str,
                log_column_new: str,
                mult_col_1: str,
                mult_col_2: str,
                mult_new_col: str,
                sub_col_1: str,
                sub_col_2: str,
                sub_new_col: str,
                nr_col_1: str,
                nr_col_2: str,
                nr_col_3: str,
                nr_col_new: str
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This function perform feature engineering on several features
        (entropy, visible_entropy, IR_range, IR_norm_range')


    Args:
        input_path (str): path to the cleaned dataset.
        columns (List): useful columns.
        log_column (str): the column that to take log
        log_column_new (str): the column name after taking log
        mult_col_1 (str): the first column that will need to take multiplication
        mult_col_2 (str): the second column that will need to take multiplication
        mult_new_col (str): the result of the multipication
        sub_col_1 (str): The name of the minuend column.
        sub_col_2 (str): The name of the subtrahend column
        sub_new_col (str): The name of the subtraction result column.
        nr_col_1 (str): The name of the minuend column
        nr_col_2 (str): The name of the subtrahend column
        nr_col_3 (str): The name of the third column (divisor)
        nr_col_new (str): The name of the resulting column.


    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    # validate and read in csv files
    if not isinstance(input_path, str):
        logger.error("The input input_path is not a string")
        raise TypeError("Input input_path expecting a string as a value, not %s." % str({type(input_path)}))
    data = csv_in(input_path=input_path)

    # check if all elements in the list column exist in column of the read in data.
    all_cols = data.columns.values.tolist()
    if all(elem in all_cols for elem in columns):
        features = data[columns]
    else:
        logger.error("Some columns in the columns input is not in the dataset. Check again!")
        raise KeyError("Some columns in the columns input is not in the dataset. Check again!")

    # start feature engineering
    target = pd.DataFrame(data["class"])

    features = take_log(df=features, old_column=log_column, new_column_name=log_column_new)
    features = multiplication(df=features, old_column1=mult_col_1, old_column2=mult_col_2,
                              new_column_name= mult_new_col)
    features = subtraction(df=features, old_column1=sub_col_1, old_column2=sub_col_2,
                           new_column_name=sub_new_col)
    features = norm_range(df=features, old_column1=nr_col_1, old_column2=nr_col_2,
                          old_column3=nr_col_3, new_column_name= nr_col_new)

    return target, features
