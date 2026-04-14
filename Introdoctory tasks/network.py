import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_layers=[100, 100], activation=nn.Tanh):
        """
        input_size: int, number of features in input
        output_size: int, number of output neurons
        hidden_layers: list of ints, each int is the number of neurons in that hidden layer
        activation: nn.Module, activation function class (default nn.Tanh)
        """
        super(MLP, self).__init__()

        layers = []

        # Input layer → first hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Combine layers into a Sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)