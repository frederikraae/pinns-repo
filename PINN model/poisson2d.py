import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import PINN
from torch import sin, exp, pi

torch.manual_seed(1)

l = 10

# u = 1/6 * (x**3*y + x*y**3)
# lap u = 2*x*y

# Chosen u(x,y) = y*cos(x) + x*cos(y), f(x) = u''(x) = -(y*cos(x) + x*cos(y))
u_chos = lambda x, y: y * torch.cos(x) + x * torch.cos(y)
f = lambda x, y: - (y * torch.cos(x) + x * torch.cos(y))

# u_chos = lambda x, y: 1/6 * (x**3*y + x*y**3)
# f = lambda x, y: 2*x*y

# Boundary points
N_b = 100

y = l * torch.rand(N_b, 1)
x = torch.zeros_like(y)
X_left = torch.cat([x,y], dim=1) # x = 0, y ∈ [0, l]

x = l * torch.ones_like(y)
X_right = torch.cat([x,y], dim=1) # x = l, y ∈ [0, l]

x = l * torch.rand(N_b, 1)
y = torch.zeros_like(x)
X_bottom = torch.cat([x,y], dim=1) # x ∈ [0, l], y = 0

y = l * torch.ones_like(x)
X_top = torch.cat([x,y], dim=1) # x ∈ [0, l], y = l

X_boundary = torch.cat([X_left, X_right, X_bottom, X_top], dim=0)

# Initialize network and define optimizer
net = PINN(
    in_dim=2,
    out_dim=1,
    hidden_dim=32,
    hidden_layers=4
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

lam_bc = 10.0

# Training loop
n_epoch = 10_000
# Interior points
N = 800

loss_history = []
loss_pde_history = []
loss_bc_history = []

for epoch in range(n_epoch):
    optimizer.zero_grad()

    x = l * torch.rand(N, 1)
    y = l * torch.rand(N, 1)
    X = torch.cat([x,y], dim=1)

    X.requires_grad_(True)

    u = net(X)

    # Computes Laplacian of u
    grad_u = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:,0:1]
    u_y = grad_u[:,1:2]

    u_xx = torch.autograd.grad(
        u_x, X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:,0:1]

    u_yy = torch.autograd.grad(
        u_y, X,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:,1:2]

    laplace_u = u_xx + u_yy

    # PDE loss
    f_val = f(X[:, 0:1], X[:, 1:2])
    loss_pde = torch.mean((laplace_u - f_val)**2)

    # BC loss
    u_bc = net(X_boundary)
    u_true_bc = u_chos(X_boundary[:,0:1], X_boundary[:,1:2])
    loss_bc = torch.mean((u_bc - u_true_bc)**2)

    # Total loss
    loss = loss_pde + lam_bc * loss_bc

    loss_history.append(loss.item())
    loss_pde_history.append(loss_pde.item())
    loss_bc_history.append(loss_bc.item())

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
n_test = 1000

x = torch.linspace(0, l, n_test)
y = torch.linspace(0, l, n_test)

X, Y = torch.meshgrid(x, y, indexing="ij")

XY = torch.cat(
    [X.reshape(-1,1), Y.reshape(-1,1)],
    dim=1
)

net.eval()

with torch.no_grad():
    u_pred = net(XY)
    u_exact = u_chos(XY[:,0:1], XY[:,1:2])

u_pred = u_pred.reshape(n_test, n_test)
u_exact = u_exact.reshape(n_test, n_test)

error = u_pred - u_exact

# Convert to numpy for plotting
Xn = X.numpy()
Yn = Y.numpy()
u_pred_n = u_pred.numpy()
u_exact_n = u_exact.numpy()
error_n = error.numpy()

# Use same levels for exact and prediction
vmin = min(u_exact_n.min(), u_pred_n.min())
vmax = max(u_exact_n.max(), u_pred_n.max())
levels = np.linspace(vmin, vmax, 30)

# Separate symmetric scale for error
err_max = np.max(np.abs(error_n))
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Exact
cf1 = axes[0].contourf(Xn, Yn, u_exact_n, levels=levels)
axes[0].set_title("Exact solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cf1, ax=axes[0])

# Prediction
cf2 = axes[1].contourf(Xn, Yn, u_pred_n, levels=levels)
axes[1].set_title("PINN prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cf2, ax=axes[1])

# Error
from matplotlib.colors import TwoSlopeNorm

# Force 0 to be the center color
err_norm = TwoSlopeNorm(
    vmin=-err_max,
    vcenter=0.0,
    vmax=err_max
)

cf3 = axes[2].contourf(
    Xn, Yn, error_n,
    levels=30,
    cmap="coolwarm",
    norm=err_norm
)

axes[2].set_title("Error: prediction - exact")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(cf3, ax=axes[2])

plt.show()

plt.figure(figsize=(8, 5))

plt.semilogy(loss_history, label="Total loss")
plt.semilogy(loss_pde_history, label="PDE loss")
plt.semilogy(loss_bc_history, label="BC loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.legend()
plt.grid(True)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Exact solution (blue)
# ax.plot_surface(
#     X.numpy(),
#     Y.numpy(),
#     u_exact.numpy(),
#     color="blue",
#     alpha=0.7,
#     linewidth=0
# )

# # PINN prediction (red)
# ax.plot_surface(
#     X.numpy(),
#     Y.numpy(),
#     u_pred.numpy(),
#     color="red",
#     alpha=0.7,
#     linewidth=0
# )

# ax.set_title("PINN vs Exact Solution")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("u(x,y)")

# plt.show()