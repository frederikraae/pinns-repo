import torch
import torch.nn as nn
import torch.nn.init as init

class Expert(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, hidden_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        gain = init.calculate_gain("tanh")

        for layer in self.hidden_layers:
            init.xavier_uniform_(layer.weight, gain=gain)
            init.zeros_(layer.bias)
        
        init.xavier_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self.output_layer(x)


class GatingNet(nn.Module):
    def __init__(
            self, 
            in_dim, 
            num_experts,
            hidden_dim=32,
            hidden_layers=2,
            temperature=1.0,
            init_mode="uniform"
            ):
        super().__init__()

        self.temperature = temperature
        self.init_mode = init_mode

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, num_experts)

        self.initialize_weights()

    def initialize_weights(self):
        gain = init.calculate_gain("tanh")

        for layer in self.hidden_layers:
            init.xavier_uniform_(layer.weight, gain=gain)
            init.zeros_(layer.bias)

        if self.init_mode == "uniform":
            init.zeros_(self.output_layer.weight)
            init.zeros_(self.output_layer.bias)

        elif self.init_mode == "xavier":
            init.xavier_uniform_(self.output_layer.weight)
            init.zeros_(self.output_layer.bias)

        else:
            raise ValueError("init_mode must be 'uniform' or 'xavier'.")
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        
        logits = self.output_layer(x)
        weights = torch.softmax(logits / self.temperature, dim=1)

        return weights, logits

    
class MoEPINN(nn.Module):
    def __init__(
            self, 
            in_dim,
            out_dim,
            num_experts=2,
            expert_hidden_dim=64,
            expert_hidden_layers=3,
            gate_hidden_dim=32,
            gate_hidden_layers=2,
            temperature=1.0, 
            gate_init_mode="uniform"
            ):
        super().__init__()

        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            Expert(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=expert_hidden_dim,
                hidden_layers=expert_hidden_layers
            )
            for _ in range(num_experts)
        ])

        self.gating = GatingNet(
            in_dim=in_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            hidden_layers=gate_hidden_layers,
            temperature=temperature,
            init_mode=gate_init_mode
        )

    def forward(self, x):
        gate_weights, gate_logits = self.gating(x)
        gate_weights_expanded = gate_weights.unsqueeze(-1)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        u_hat = torch.sum(gate_weights_expanded * expert_outputs, dim=1)

        return u_hat, gate_weights, gate_logits, expert_outputs
    

class Sampling():
    pass






# ========== DIAGNOSTICS ===========

def vdp_residual(model, t, mu):
    x, _, _, _ = model(t)

    x_t = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    x_tt = torch.autograd.grad(
        x_t, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    R = x_tt - mu * (1 - x**2) * x_t + x
    return R

