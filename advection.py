import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import MLP

# Choose a function h(x)
h_chos = lambda x: np.sin(x)

# Define wave speed
c = 2.5

# Define spacial range
xa = -2.0
xb = 2.0

# Define time range
t0 = 0.0
t_final = 10.0

# Define boundary points
N_bc = 100

t_bc = t_final * torch.rand(N_bc, 1)
x_bc = xa * torch.ones_like(t_bc)

X_bc = torch.cat([x_bc, t_bc], dim=1)

# Define initial points
N_ic = 100

x_ic = xa + (xb - xa) * torch.rand(N_ic, 1)
t_ic = t0 * torch.ones_like(x_ic)

X_ic = torch.cat([x_ic, t_ic], dim=1)

# Initialize network and define optimizer
net = MLP(2,1,[64,64,64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

lam_pde = 1.0
lam_bc = 1.0
lam_ic = 1.0

# Training loop
n_epoch = 1000
for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Interior points
    N = 1000

    x = xa + (xb - xa) * torch.rand(N, 1)
    t = t_final * torch.rand(N, 1)
    X = torch.cat([x,t], dim=1)

    X.requires_grad_(True)

    u = net(X)

    # Compute derivatives
    grad_u = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # PDE loss
    loss_pde = torch.mean((u_t + c * u_x)**2)

    # BC loss
    u_bc = net(X_bc)
    u_true_bc = h_chos(X_bc[:, 0:1] - c * X[X_bc[:, 1:2]])
    loss_bc = torch.mean((u_bc - u_true_bc)**2)

    # IC loss
    u_ic = net(X_ic)
    u_true_ic = h_chos(X_ic[:, 0:1], X_ic[:, 1:2])
    loss_ic = torch.mean((u_bc - u_true_bc)**2)

    # Total loss
    loss = lam_pde * loss_pde + lam_bc * loss_bc + lam_ic * loss_ic

    loss.backward()
    optimizer.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )


