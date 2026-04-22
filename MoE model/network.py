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
    def __init__(self, input_dim=1, output_dim=1, hidden=[100,100], n_experts=2, temperature=1):
        super().__init__()

        self.n_experts = n_experts
        self.temperature = temperature

        # ---- Experts ----
        self.experts = nn.ModuleList([
            MLP(
                input_size=input_dim, 
                output_size=output_dim, 
                hidden_layers=hidden
                ) for _ in range(n_experts)])

        # ---- Gating network ----
        self.gate = MLP(
            input_size=input_dim, 
            output_size=n_experts, 
            hidden_layers=[32,32]
            )

        # ---- Initialise weights (Xavier) ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # ---- Expert outputs ----
        expert_outputs = [expert(x) for expert in self.experts]   # K × (N,1)
        expert_outputs = torch.stack(expert_outputs, dim=-1)      # (N,1,K)

        # ---- Gating weights ----
        gate_logits = self.gate(x)                                # (N,K)
        gate_weights = torch.softmax(gate_logits / self.temperature, dim=1)          # (N,K)
        gate_weights = gate_weights.unsqueeze(1)                  # (N,1,K)

        # ---- Mixture ----
        u_hat = torch.sum(gate_weights * expert_outputs, dim=-1)  # (N,1)

        return u_hat, gate_weights
    
class SoftAdapt:
    def __init__(self, beta=0.1, eps=1e-8, Normalized=False, Loss_weighted=False):
        self.beta = beta
        self.eps = eps
        self.f_prev = None
        self.Loss_weighted = Loss_weighted
        self.Normalized = Normalized


    def get_alphas(self, losses):
        """
        losses: list of scalar torch tensors
        returns: tensor of weights summing to 1
        """
        # current loss values (no grad)
        f_curr = torch.tensor([l.detach().item() for l in losses], dtype=torch.float32)

        # first call => equal weights
        if self.f_prev is None:
            self.f_prev = f_curr.clone()
            return torch.ones_like(f_curr) / len(f_curr)
        
        # rate of change
        s = f_curr - self.f_prev
        self.f_prev = f_curr.clone()

        if self.Normalized:
            s = s / (torch.sum(torch.abs(s)) + self.eps)

        # --- numerically stable softmax ---
        s_max = torch.max(s)
        exp_s = torch.exp(self.beta * (s - s_max))
        alphas = exp_s / (torch.sum(exp_s) + self.eps)

        if self.Loss_weighted:
            alphas = alphas * f_curr / (torch.sum(alphas * f_curr) + self.eps)

        return alphas