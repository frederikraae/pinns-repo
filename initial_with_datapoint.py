#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import MLP

#%%
# Setup intervals
a_lam, b_lam = -1, 1
a_t, b_t = 0, 20

# Initial values
u0 = 1
N_i = 100

lam_ic = a_lam + (b_lam - a_lam) * torch.rand(N_i, 1)
t_ic = torch.zeros_like(lam_ic)
X_ic = torch.cat([t_ic,lam_ic], dim=1)

# Datapoint
t_d = torch.tensor([[6.0]])
lam_d = torch.tensor([[-0.5]])
X_d = torch.cat([t_d, lam_d], dim=1)

u_d_true = torch.exp(lam_d * t_d)

# Initialize network and define optimizer
net = MLP(2,1,[64,64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

w_ic = 1.0
w_data = 1.0

# Traing loop
n_epoch = 5000
for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Interior points
    N = 1000
    
    t = a_t + (b_t - a_t) * torch.rand(N, 1)
    lam = a_lam + (b_lam - a_lam) * torch.rand(N, 1)
    X = torch.cat([t, lam], dim=1)

    X.requires_grad_()

    u = net(X)

    # Computes the gradient of u
    grad_u = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_t = grad_u[:, 0:1]

    # PDE loss
    loss_pde = torch.mean((u_t - lam * u)**2)

    # IC loss
    u_ic = net(X_ic)
    loss_ic = torch.mean((u_ic - u0)**2)

    # Datapoint loss
    u_d_pred = net(X_d)
    loss_data = torch.mean((u_d_pred - u_d_true)**2)

    # Total loss
    loss = loss_pde + w_ic * loss_ic + w_data * loss_data

    loss.backward()
    optimizer.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e} "
            f"data={loss_data.item():.6e}"
        )

#%%
# Evaluation
N = 200
lam_chos = -0.8

t_eval = torch.linspace(0, 20, N).reshape(-1, 1)
lam_eval = lam_chos * torch.ones_like(t_eval)

X_eval = torch.cat([t_eval, lam_eval], dim=1)

net.eval()
with torch.no_grad():
    u_pred = net(X_eval)

u_exact = torch.exp(lam_chos * t_eval)

#%%
plt.figure()

plt.plot(t_eval.numpy(), u_pred.numpy(), label="PINN")
plt.plot(t_eval.numpy(), u_exact.numpy(), "--", label="Exact")
# Plot datapoint
# plt.scatter(t_d, u_d_true, color="red", s=40, label="Datapoint")

plt.xlabel("t")
plt.ylabel(f"u(t, λ={lam_chos})")
plt.title(f"PINN vs Exact Solution for λ = {lam_chos}")

plt.legend()
plt.show()

# %%
