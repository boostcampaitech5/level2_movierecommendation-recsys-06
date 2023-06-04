import torch
import torch.nn as nn
import models
import utill.get_model_path as get_model_path


def load_model(setting: dict, is_train: bool = True) -> nn.Module:
    """Load a Model

    Args:
        setting (dict): Required Settings Dictionary
        is_train (bool): if train = Create model object ,
                         else = Load trained model

    Raises:
        NameError: Occurs when the selected model does not exist

    Returns:
        nn.Module: Return model
    """
    model_name = setting["model_name"].lower()

    if model_name == "mlp":
        model: nn.Module = MultiLayerPerceptron(setting=setting)
    else:
        raise NameError(f"Not Found model : {model_name}")

    if not is_train:
        model_path = get_model_path()
        model.load_state_dict(torch.load(model_path).to(setting["device"]))

    return model.to(setting["device"])
