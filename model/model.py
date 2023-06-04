import torch
import torch.nn as nn
import models
import utill.get_model_path as get_model_path


def create_model(setting: dict) -> nn.Module:
    """Creating a Model Object

    Args:
        setting (dict): Required Settings Dictionary

    Raises:
        NameError: Occurs when the selected model does not exist

    Returns:
        nn.Module: Return model
    """
    model_name = setting["model_name"].lower()

    if model_name == "mlp":
        model: nn.Module = MultiLayerPerceptron(setting=setting)
    else:
        raise NameError("Not Found model : {model_name}")

    return model.to(setting["device"])


def load_model(setting: dict) -> nn.Module:
    """Recall Trained Models

    Args:
        setting (dict): Required Settings Dictionary

    Raises:
        NameError: Occurs when the selected model does not exist

    Returns:
        nn.Module: Return Trained model
    """
    model_name = setting["model_name"].lower()
    model_path = get_model_path()

    if model_name == "mlp":
        model: nn.Module = MultiLayerPerceptron(setting=setting)
    else:
        raise NameError("Not Found model : {model_name}")

    model.load_state_dict(torch.load(model_path).to(setting["device"]))

    return model
