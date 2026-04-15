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
    
class MoEPINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden=[64,64,64], n_experts=2):
        super().__init__()
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            MLP(input_dim, output_dim, hidden) for _ in range(n_experts)
        ])

        self.gate = MLP(input_dim, n_experts, [64, 64])

    def forward(self, x):
        # x has shape (N, 2) = [t, lambda]
        expert_outputs = [expert(x) for expert in self.experts]   # K tensors of shape (N,1)
        expert_outputs = torch.stack(expert_outputs, dim=-1)      # (N,1,K)

        gate_logits = self.gate(x)                                # (N,K)
        gate_weights = torch.softmax(gate_logits, dim=1)          # (N,K)
        gate_weights = gate_weights.unsqueeze(1)                  # (N,1,K)

        u_hat = torch.sum(gate_weights * expert_outputs, dim=-1)  # (N,1)
        return u_hat, gate_weights