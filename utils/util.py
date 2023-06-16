from datetime import datetime
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
    Gets raw data from pickle file.

    Parameters:
        settings(dict): Dictionary containing settings
    Returns:
        raw_user_df(pd.DataFrame): Raw user data
        raw_item_df(pd.DataFrame): Raw item data
        raw_interaction_df(pd.DataFrame): Raw interaction data
    """

    data_path: str = settings["path"]["data"]
    df_path: str = settings["input_data"]

    raw_interaction_df: pd.DataFrame = pd.read_pickle(
        os.path.join(data_path, df_path[0])
    )
    raw_user_df: pd.DataFrame = pd.read_pickle(os.path.join(data_path, df_path[1]))
    raw_item_df: pd.DataFrame = pd.read_pickle(os.path.join(data_path, df_path[2]))

    return raw_user_df, raw_item_df, raw_interaction_df


def create_data(settings) -> tuple:
    """
    Creates pickle file from initial input data

    Parameters:
        settings(dict): Dictionary containing settings
    Returns:
        user_df(pd.DataFrame): Raw user data
        item_df(pd.DataFrame): Raw item data
        interaction_df(pd.DataFrame): Raw interaction data
    """

    data_path = settings["path"]["data"]

    input_df = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    input_df.rename(
        columns={"user": "user_id", "item": "item_id", "time": "unix_time"},
        inplace=True,
    )
    input_df["date_actual"] = input_df["unix_time"].apply(
        lambda x: datetime.utcfromtimestamp(x).strftime("%Y%m%d%H%M%S")
    )
    input_df["date_year"] = input_df.date_actual.apply(lambda x: x[:4])
    input_df["date_month"] = input_df.date_actual.apply(lambda x: x[4:6])
    input_df["date_day"] = input_df.date_actual.apply(lambda x: x[6:8])
    input_df["date_hour"] = input_df.date_actual.apply(lambda x: x[8:10])
    input_df["date_minute"] = input_df.date_actual.apply(lambda x: x[10:12])
    input_df["date_second"] = input_df.date_actual.apply(lambda x: x[12:14])

    input_df.to_pickle(os.path.join(settings["path"]["data"], "input_data.pkl"))

    user_df = pd.DataFrame(sorted(input_df["user_id"].unique()), columns=["user_id"])
    user_df["item_watched_num"] = user_df["user_id"].map(
        input_df.groupby("user_id")["item_id"].count()
    )

    user_df.to_pickle(os.path.join(settings["path"]["data"], "user_data.pkl"))

    item_df = pd.DataFrame(sorted(input_df["item_id"].unique()), columns=["item_id"])
    item_df["user_watched_num"] = item_df["item_id"].map(
        input_df.groupby("item_id")["user_id"].count()
    )

    tsv_list = ["directors", "genres", "titles", "writers", "years"]
    not_idx_list = ["titles", "years"]

    indexing_dict = dict()

    for tsv in tsv_list:
        tsv_file = pd.read_csv(os.path.join(data_path, tsv + ".tsv"), sep="\t")

        if tsv not in not_idx_list:
            indexing_dict[tsv] = {
                v: i for i, v in enumerate(sorted(tsv_file[tsv[:-1]].unique()))
            }
            tsv_file[tsv[:-1]] = tsv_file[tsv[:-1]].map(indexing_dict[tsv])

        grouped_file = tsv_file.groupby("item").apply(
            lambda x: x[tsv[:-1]].values.squeeze().tolist()
        )
        if tsv not in not_idx_list:
            grouped_file = grouped_file.apply(lambda x: [x] if type(x) == int else x)

        item_df["item_" + tsv[:-1]] = item_df["item_id"].map(grouped_file)

    item_df.to_pickle(os.path.join(settings["path"]["data"], "item_data.pkl"))

    interaction_df = input_df.merge(item_df, on="item_id")
    interaction_df = interaction_df.merge(user_df, on="user_id")

    interaction_df.to_pickle(os.path.join(settings["path"]["data"], "total_data.pkl"))

    return user_df, item_df, interaction_df
