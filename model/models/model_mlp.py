import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, setting: dict):
        super(MultiLayerPerceptron, self).__init__()
        input_dim = settings["mlp"]["input_dim"]
        hidden_dims = setting["mlp"]["hidden_dims"]
        output_dim = setting["mlp"]["output_dim"]  # == The number of movie? #

        self.sequential_linear = nn.Sequential()

        for hidden_dim in self.hidden_dims:
            self.sequential_linear.append(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            input_dim = hidden_dim

        self.sequential_linear.append(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forword(x):
        output = self.sequential_linear(x)
        return output
