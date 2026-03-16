import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import MLP

# Define parameters
t0 = 0.0
t_final = 10.0
lam = 0.5
w0 = 3

# Define initial conditions
x_t0_true = torch.tensor([[1.0]])
v_t0_true = torch.tensor([[0.0]])

# Initialize network and define optimizer
net = MLP(1,1,[64,64,64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

w_ic = 1.0

# Training loop
n_epoch = 1000
for epoch in range(n_epoch):
    optimizer.zero_grad()

    N = 1000
    t = t0 + (t_final - t0) * torch.rand(N, 1)
    t.requires_grad_(True)

    x = net(t)

    # Compute velocity and acceleration
    v = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    a = torch.autograd.grad(
        v, t,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    # PDE loss
    residual = a + w0**2 * x + lam * v
    loss_pde = torch.mean(residual**2)

    # IC loss
    t0_tensor = torch.tensor([[t0]], requires_grad=True)
    x_t0 = net(t0_tensor)
    v_t0 = torch.autograd.grad(
        x_t0, t0_tensor,
        grad_outputs=torch.ones_like(x_t0),
        create_graph=True
    )[0]

    loss_ic = (x_t0 - x_t0_true)**2 + (v_t0 - v_t0_true)**2
    loss_ic = torch.mean(loss_ic)

    # Total loss
    loss = loss_pde + w_ic * loss_ic

    loss.backward()
    optimizer.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )

# Evaluation
net.eval()
t_test = torch.linspace(t0, t_final, 10000).unsqueeze(1)

with torch.no_grad():
    x_pred = net(t_test)
plt.figure()
plt.plot(t_test.numpy(), x_pred.numpy(), label="PINN")
# plt.plot(t_test.numpy(), x_exact.numpy(), "--", label="Exact solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.title("Damped Harmonic Oscillator - PINN")
plt.show()

