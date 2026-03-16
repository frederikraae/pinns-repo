import numpy as np
import torch
import torch.nn as nn   
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


class BoundaryConditionedNet(nn.Module):
    def __init__(self, L, c, d, hidden=100):
        super().__init__()
    
        self.hidden = nn.Linear(1, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

        dtype = torch.get_default_dtype()
        self.register_buffer("L", torch.tensor([[L]], dtype=dtype))
        self.register_buffer("c", torch.tensor([[c]], dtype=dtype))
        self.register_buffer("d", torch.tensor([[d]], dtype=dtype))

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x



# Define condtions
L = 2.0
c = 1.0
d = torch.e ** 2

# Initiate network and define the optimizer
net = BoundaryConditionedNet(L=L, c=c, d=d)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

lam_bc = 5.0

# Training loop
n_epochs = 5000

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Generate input data points
    x = torch.unsqueeze(torch.linspace(0, L, 1000), dim=1)
    x.requires_grad_(True)

    u = net(x)

    # Compute the second derivative (u'')
    u_x = torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True, 
        )[0]
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
        )[0]

    # PDE residual: u'' - u
    residual = u_xx - u
    loss_pde = torch.mean(residual ** 2)

    # Boundary points
    x0 = torch.tensor([[0.0]])
    xL = torch.tensor([[L]])

    u0 = net(x0)
    uL = net(xL)

    loss_bc = ((u0 - c)**2 + (uL - d)**2).mean()

    # Total loss    
    loss = loss_pde + lam_bc * loss_bc
    
    loss.backward()
    optimizer.step()

    if epoch % (n_epochs/10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e}"
        )


# Evaluation
net.eval()
x_test = torch.linspace(0, L, 200).unsqueeze(1)

with torch.no_grad():
    u_pred = net(x_test)
    u_exact = torch.exp(x_test)

plt.figure()
plt.plot(x_test.numpy(), u_pred.numpy(), label="PINN")
plt.plot(x_test.numpy(), u_exact.numpy(), "--", label="Exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Soft boundary constraint PINN")
plt.show()