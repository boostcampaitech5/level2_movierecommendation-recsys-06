import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, setting: dict):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = setting["mlp"]["input_dim"]
        self.hidden_dims = setting["mlp"]["hidden_dims"]
        self.output_dim = setting["mlp"]["output_dim"]  # == The number of movie? #

        self.module_list = nn.ModuleList()

        temp_layer = self.input_dim

        for hidden_dim in self.hidden_dims:
            self.module_list.append(nn.Linear(temp_layer, hidden_dim))
            self.module_list.append(nn.ReLU())
            temp_layer = hidden_dim

        self.module_list.append(nn.Linear(temp_layer, self.output_dim))
        self.module_list.append(nn.Sigmoid())

        self.sequential_linear = nn.Sequential(*self.module_list)

    def forword(x):
        output = self.sequential_linear(x)
        return output
