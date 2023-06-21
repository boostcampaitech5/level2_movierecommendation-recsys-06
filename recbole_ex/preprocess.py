import pickle as plk
import os
import pandas as pd
from datetime import datetime
import yaml


def load_setting(path):
    with open(path) as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)
    return setting


def load_data(data_path):
    data = dict()
    data_name_list = ["input_data", "item_data", "user_data"]
    for data_name in data_name_list:
        data[data_name] = pd.read_pickle(os.path.join(data_path, data_name + ".pkl"))
    return data


def preprocess(data: dict, save_path: str, is_saving: bool = False):
    # input data
    data = data["input_data"]
    data = data[["user_id", "item_id", "unix_time"]].rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "unix_time": "timestamp:float",
        }
    )
    if is_saving:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        dataset_name = save_path.split("/")[-1]
        data.to_csv(
            os.path.join(save_path, f"{dataset_name}.inter"), index=False, sep="\t"
        )

    return data


def merge_yaml(model_property_path, data_property_path, train_eval_path, save_path):
    with open(model_property_path) as f:
        model_yaml = yaml.load(f, Loader=yaml.FullLoader)

    with open(data_property_path) as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)

    with open(train_eval_path) as f:
        train_eval_yaml = yaml.load(f, Loader=yaml.FullLoader)

    data_yaml.update(model_yaml)
    data_yaml.update(train_eval_yaml)

    with open(save_path, "w") as f:
        yaml.dump(data_yaml, f)


if __name__ == "__main__":
    path = "./settings.yaml"

    setting = load_setting(path)

    merge_yaml(
        setting["path"]["model_property"],
        setting["path"]["data_property"],
        setting["path"]["train_valid_property"],
        setting["path"]["save_yaml"],
    )

    data = load_data(setting["path"]["data"])

    data = preprocess(data, setting["path"]["save_data"], True)

    print(data)
