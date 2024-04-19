import logging.config
import pandas as pd

logger = logging.getLogger(__name__)
# pylint: disable=no-member

def csv_in(input_path: str) -> pd.DataFrame:
    """This functino will read csv from the provided file and provide logging info.

    Args:
        input_path (str): input path.

    Returns:
        pd.DataFrame: the pandas df readed from the csv.
    """
    # check input
    try:
        df = pd.read_csv(input_path)
        logger.info("The dataset path %s is loaded and it has %i columns and %i rows.",
                      input_path, df.shape[1], df.shape[0])
    except FileNotFoundError:
        logger.error("Cannot find %s", input_path)

    return df

def csv_out(df: pd.DataFrame,
            output_path: str,
            display_str: str) -> None:
    """This function will save a df to a csv and display a logging message.

    Args:
        df (pd.DataFrame): the df that need to be saved
        output_path (str): the path to save this df
        display_str (str): the name of the dataframe (for logging message).
    """
    # check input
    if not isinstance(output_path, str):
        logger.error("The output path is not a string, not %s.", str({type(output_path)}))
        raise TypeError("Output path should be a string")

    if not isinstance(df, pd.DataFrame):
        logger.error("The input df that user intended to save at %s is not a pandas dataframe. Please check on that.",
                    output_path)
        raise TypeError("Input df for csv_out should be a pd dataframe")

    df.to_csv(output_path, index=False)
    logger.info("The dataframe %s is saved to %s", display_str, output_path)
