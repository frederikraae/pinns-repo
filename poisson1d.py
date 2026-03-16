import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import MLP

# Chosen u(x) = cos(x), f(x) = u''(x) = -cos(x)
u_chos = lambda x: torch.cos(x)
f = lambda x: -torch.cos(x)

# Boundary points
a = 0.0
b = 3 * torch.pi

# Initialize network and define optimizer
net = MLP(1,1,[64,64,64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

lam_bc = 1.0

# Training loop
n_epoch = 1000
for epoch in range(n_epoch):
    optimizer.zero_grad()

    x = torch.unsqueeze(torch.linspace(a,b,100), dim=1)
    x.requires_grad_(True)

    u = net(x)

    # Compute laplacian of u
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # PDE loss
    residual = u_xx - f(x)
    loss_pde = torch.mean(residual**2)

    # BC loss
    x_a = torch.tensor([[a]])
    x_b = torch.tensor([[b]])

    u_a = net(x_a)
    u_b = net(x_b)

    loss_bc = (u_a - u_chos(x_a))**2 + (u_b - u_chos(x_b))**2
    loss_bc = loss_bc.mean()

    # Total loss
    loss = loss_pde + lam_bc * loss_bc

    loss.backward()
    optimizer.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e}"
        )


# Evaluation
net.eval()
x_test = torch.linspace(a, b, 200).unsqueeze(1)

with torch.no_grad():
    u_pred = net(x_test)
    u_exact = torch.cos(x_test)

plt.figure()
plt.plot(x_test.numpy(), u_pred.numpy(), label="PINN")
plt.plot(x_test.numpy(), u_exact.numpy(), "--", label="Exact solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("1D Poisson - PINN")
plt.show()