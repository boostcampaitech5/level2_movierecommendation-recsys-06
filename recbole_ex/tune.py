import math

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from util import load_setting

FIXED_HP_PATH = "/opt/ml/level2_movierecommendation-recsys-06/recbole_ex/fixed_hp.yaml"
TUNE_HP_PATH = "/opt/ml/level2_movierecommendation-recsys-06/recbole_ex/tune_hp.hyper"


def ray_tune(setting):
    ray.init()
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(TUNE_HP_PATH, "r") as fp:
        for line in fp:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.choice(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(
                    math.exp(float(low)), math.exp(float(high))
                )
            else:
                raise ValueError("Illegal param type [{}]".format(para_type))
    scheduler = ASHAScheduler(
        metric="recall@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    local_dir = "./ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=[FIXED_HP_PATH]),
        config=config,
        num_samples=5,
        log_to_file="tune.output",
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"gpu": 1},
    )

    best_trial = result.get_best_trial("recall@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


if __name__ == "__main__":
    setting = load_setting(FIXED_HP_PATH)

    if setting["tool"] == "ray":
        ray_tune(setting)
