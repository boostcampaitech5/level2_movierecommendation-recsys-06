import pickle as plk
import os
import pandas as pd
from datetime import datetime
import yaml
import logging
from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger


def load_setting(path):
    with open(path) as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)
    return setting


# def load_data(data_path):
#     data = dict()
#     data_name_list = ["input_data", "item_data", "user_data"]
#     for data_name in data_name_list:
#         data[data_name] = pd.read_pickle(os.path.join(data_path, data_name + ".pkl"))
#     return data

def load_data(data_path):
    data = dict()
    data_name_list = ["inter", "item", "user"]
    for data_name in data_name_list:
        data[data_name] = pd.read_csv(os.path.join(data_path,data_name+"_df."+data_name), sep='\t')
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
    return data_yaml


def get_config(model, dataset, config_dict):
    config = Config(model=model, dataset=dataset, config_dict=config_dict)

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # write config info into log
    logger.info(config)

    return logger, config


if __name__ == "__main__":
    setting = load_setting(PATH)

    merge_yaml(
        setting["path"]["model_property"],
        setting["path"]["data_property"],
        setting["path"]["train_valid_property"],
        setting["path"]["save_yaml"],
    )

    data = load_data(setting["path"]["data"])
