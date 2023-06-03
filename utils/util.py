import os
import pandas as pd
import yaml


def get_settings() -> dict:
    """
    Imports settings from yaml file.

    Returns:
        setting_dict(dict): Dictionary containing settings
    """

    with open("config.yaml", "r") as f:
        setting_dict: dict = yaml.safe_load(f)

    return setting_dict


def get_raw_data(settings) -> pd.DataFrame:
    """
    Gets raw data from csv file.

    Parameters:
        settings(dict): Contains settings
    Returns:
        raw_input_df(pd.DataFrame): Raw data file
    """

    data_path: str = settings["path"]["data"]
    df_path: str = settings["input_data"]

    raw_input_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, df_path))

    return raw_input_df
