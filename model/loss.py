import torch
import losses
from torch.nn import MSELoss


def get_loss_fn(setting: dict):
    """Get loss function

    Args:
        setting (dict): Required Settings Dictionary

    Raises:
        NameError: Behavior when no loss_function is selected

    Returns:
        loss_function : return loss function, for example RMSE, MSE
    """

    loss_fn_name = setting["loss_fn"].lower()
    if loss_fn_name == "rmse":
        loss_fn = RMSELoss()
    elif loss_fn_name == "mse":
        loss_fn = MSELoss()
    else:
        raise NameError(f"Not found loss function : {loss_fn_name}")

    return loss_fn
