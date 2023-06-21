from recbole.quick_start import run_recbole
from preprocess import preprocess
from util import load_setting, merge_yaml, load_data

PATH = "/opt/ml/movie_rec_project/level2_movierecommendation-recsys-06/recbole_ex/settings.yaml"


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

    # load data
    data = load_data(setting["path"]["data"])

    # preprocess
    data = preprocess(data, setting["path"]["save_data"], False)

    # Get logger and config
    run_recbole(model="GRU4Rec", dataset="recbole_data", config_dict=data_yaml)


if __name__ == "__main__":
    main()
