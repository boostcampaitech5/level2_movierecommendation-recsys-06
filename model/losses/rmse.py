import torch
import torch.nn as nn
from torch.nn import MSELoss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, y, y_hat):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(y, y_hat) + self.eps)
        return loss
