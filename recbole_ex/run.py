from recbole.quick_start import run_recbole
from preprocess import preprocess
from util import load_setting, merge_yaml, load_data

PATH = "./settings.yaml"


def main():
    # load setting
    setting = load_setting(PATH)

    # merge property_yaml
    data_yaml = merge_yaml(
        setting["path"]["model_property"],
        setting["path"]["data_property"],
        setting["path"]["train_valid_property"],
        setting["path"]["save_yaml"],
    )

    # # load data
    # data = load_data(setting["path"]["data"])

    # # preprocess
    # data = preprocess(data, setting["path"]["save_data"], False)

    # Get logger and config
    run_recbole(model="NCL", dataset="datav4", config_dict=data_yaml)


if __name__ == "__main__":
    main()
