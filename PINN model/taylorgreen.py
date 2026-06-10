#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from network import PINN
from torch import sin, cos, exp, pi
from time import perf_counter

import sys

start = perf_counter()

torch.manual_seed(1)

# Parameters
V0 = 1.0
L = 1.0
rho = 1.0
nu = 0.01

T = 10

# Analytical solution
u_analytical = lambda x, y, t: V0 * cos(x / L) * sin(y / L) * exp(-2 * (nu / L**2) * t)
v_analytical = lambda x, y, t: -V0 * sin(x / L) * cos(y / L) * exp(-2 * (nu / L**2) * t)
p_analytical = lambda x, y, t: (-rho / 4) * V0**2 * (cos(2*x / L) + cos(2*y / L)) * exp(-4 * (nu / L**2) * t)

# Initialize network and define optimizer
net = PINN(
    in_dim=3,
    out_dim=3,
    hidden_dim=128,
    hidden_layers=8
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Initial points
N_ic = 2000
x_ic = - (L * pi) + 2 * L * pi * torch.rand(N_ic, 1)
y_ic = - (L * pi) + 2 * L * pi * torch.rand(N_ic, 1)
t_ic = torch.zeros_like(x_ic)
X_ic = torch.cat([x_ic,y_ic,t_ic], dim=1)


# Loss term weights
lam_pde = 3.0
lam_bc = 5.0
lam_ic = 5.0

# Interior points (collacation)
N = int(sys.argv[1])

loss_history = []
loss_pde_history = []
loss_bc_history = []
loss_ic_history = []

#%%
# Training loop
n_epoch = 15_000
for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Collocation points
    x = - (L * pi) + 2 * L * pi * torch.rand(N, 1)
    y = - (L * pi) + 2 * L * pi * torch.rand(N, 1)
    t = T * torch.rand(N, 1)
    X = torch.cat([x,y,t], dim=1)
    
    X.requires_grad_(True)

    # Boundary points
    pi_like = pi * torch.ones_like(x)
    X_bc_left = torch.cat([-pi_like,y,t], dim=1)
    X_bc_right = torch.cat([pi_like,y,t], dim=1)
    X_bc_bottom = torch.cat([x,-pi_like,t], dim=1)
    X_bc_top = torch.cat([x,pi_like,t], dim=1)

    X_bc_left.requires_grad_(True)
    X_bc_right.requires_grad_(True)
    X_bc_bottom.requires_grad_(True)
    X_bc_top.requires_grad_(True)

    # Forward pass
    G = net(X)

    u = G[:, 0:1]
    v = G[:, 1:2]
    p = G[:, 2:3]

    grad_u = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    grad_v = torch.autograd.grad(
        v, X,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    v_t = grad_v[:, 2:3]

    grad_p = torch.autograd.grad(
        p, X,
        grad_outputs=torch.ones_like(p),
        create_graph=True
    )[0]

    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]


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

    v_xx = torch.autograd.grad(
        v_x, X,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True
    )[0][:,0:1]

    v_yy = torch.autograd.grad(
        v_y, X,
        grad_outputs=torch.ones_like(v_y),
        create_graph=True
    )[0][:,1:2]


    # PDE loss
    loss_pde_eq1 = torch.mean((u_x + v_y)**2)
    loss_pde_eq2 = torch.mean((u_t + u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy))**2) 
    loss_pde_eq3 = torch.mean((v_t + u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy))**2)
    
    loss_pde = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3

    # BC loss
    G_bc_left = net(X_bc_left)
    G_bc_right = net(X_bc_right)
    G_bc_bottom = net(X_bc_bottom)
    G_bc_top = net(X_bc_top)

    u_bc_left = G_bc_left[:, 0:1]
    v_bc_left = G_bc_left[:, 1:2]
    p_bc_left = G_bc_left[:, 2:3]

    u_bc_right = G_bc_right[:, 0:1]
    v_bc_right = G_bc_right[:, 1:2]
    p_bc_right = G_bc_right[:, 2:3]

    u_bc_bottom = G_bc_bottom[:, 0:1]
    v_bc_bottom = G_bc_bottom[:, 1:2]
    p_bc_bottom = G_bc_bottom[:, 2:3]

    u_bc_top = G_bc_top[:, 0:1]
    v_bc_top = G_bc_top[:, 1:2]
    p_bc_top = G_bc_top[:, 2:3]

    loss_bc_x = torch.mean((u_bc_left - u_bc_right)**2) + torch.mean((v_bc_left - v_bc_right)**2) + torch.mean((p_bc_left - p_bc_right)**2)
    loss_bc_y = torch.mean((u_bc_bottom - u_bc_top)**2) + torch.mean((v_bc_bottom - v_bc_top)**2) + torch.mean((p_bc_bottom - p_bc_top)**2)

    grad_u_bc_left = torch.autograd.grad(
        u_bc_left, X_bc_left,
        grad_outputs=torch.ones_like(u_bc_left),
        create_graph=True
    )[0]

    u_x_bc_left = grad_u_bc_left[:, 0:1]

    grad_u_bc_right = torch.autograd.grad(
        u_bc_right, X_bc_right,
        grad_outputs=torch.ones_like(u_bc_right),
        create_graph=True
    )[0]

    u_x_bc_right = grad_u_bc_right[:, 0:1]

    grad_u_bc_bottom = torch.autograd.grad(
        u_bc_bottom, X_bc_bottom,
        grad_outputs=torch.ones_like(u_bc_bottom),
        create_graph=True
    )[0]

    u_y_bc_bottom = grad_u_bc_bottom[:, 1:2]

    grad_u_bc_top = torch.autograd.grad(
        u_bc_top, X_bc_top,
        grad_outputs=torch.ones_like(u_bc_top),
        create_graph=True
    )[0]

    u_y_bc_top = grad_u_bc_top[:, 1:2]


    grad_v_bc_left = torch.autograd.grad(
        v_bc_left, X_bc_left,
        grad_outputs=torch.ones_like(v_bc_left),
        create_graph=True
    )[0]

    v_x_bc_left = grad_v_bc_left[:, 0:1]

    grad_v_bc_right = torch.autograd.grad(
        v_bc_right, X_bc_right,
        grad_outputs=torch.ones_like(v_bc_right),
        create_graph=True
    )[0]

    v_x_bc_right = grad_v_bc_right[:, 0:1]

    grad_v_bc_bottom = torch.autograd.grad(
        v_bc_bottom, X_bc_bottom,
        grad_outputs=torch.ones_like(v_bc_bottom),
        create_graph=True
    )[0]

    v_y_bc_bottom = grad_v_bc_bottom[:, 1:2]

    grad_v_bc_top = torch.autograd.grad(
        v_bc_top, X_bc_top,
        grad_outputs=torch.ones_like(v_bc_top),
        create_graph=True
    )[0]

    v_y_bc_top = grad_v_bc_top[:, 1:2]

    grad_p_bc_left = torch.autograd.grad(
        p_bc_left, X_bc_left,
        grad_outputs=torch.ones_like(p_bc_left),
        create_graph=True
    )[0]

    p_x_bc_left = grad_p_bc_left[:, 0:1]

    grad_p_bc_right = torch.autograd.grad(
        p_bc_right, X_bc_right,
        grad_outputs=torch.ones_like(p_bc_right),
        create_graph=True
    )[0]

    p_x_bc_right = grad_p_bc_right[:, 0:1]

    grad_p_bc_bottom = torch.autograd.grad(
        p_bc_bottom, X_bc_bottom,
        grad_outputs=torch.ones_like(p_bc_bottom),
        create_graph=True
    )[0]

    p_y_bc_bottom = grad_p_bc_bottom[:, 1:2]

    grad_p_bc_top = torch.autograd.grad(
        p_bc_top, X_bc_top,
        grad_outputs=torch.ones_like(p_bc_top),
        create_graph=True
    )[0]

    p_y_bc_top = grad_p_bc_top[:, 1:2]

    loss_bc_grad_x = torch.mean((u_x_bc_left - u_x_bc_right)**2) + torch.mean((v_x_bc_left - v_x_bc_right)**2) + torch.mean((p_x_bc_left - p_x_bc_right)**2)
    loss_bc_grad_y = torch.mean((u_y_bc_bottom - u_y_bc_top)**2) + torch.mean((v_y_bc_bottom - v_y_bc_top)**2) + torch.mean((p_y_bc_bottom - p_y_bc_top)**2)

    loss_bc = loss_bc_x + loss_bc_y + loss_bc_grad_x + loss_bc_grad_y
   
    # IC loss
    G_ic = net(X_ic)

    u_ic = G_ic[:, 0:1]
    v_ic = G_ic[:, 1:2]
    p_ic = G_ic[:, 2:3]

    loss_ic_eq1 = torch.mean((u_ic - cos(x_ic) * sin(y_ic))**2)
    loss_ic_eq2 = torch.mean((v_ic + sin(x_ic) * cos(y_ic))**2)
    loss_ic_eq3 = torch.mean((p_ic + (1/4) * (cos(2*x_ic) + cos(2*y_ic)))**2)

    loss_ic = loss_ic_eq1 + loss_ic_eq2 + loss_ic_eq3

    # Total loss
    loss = lam_pde * loss_pde + lam_bc * loss_bc + lam_ic * loss_ic

    loss_history.append(loss.item())
    loss_pde_history.append(loss_pde.item())
    loss_bc_history.append(loss_bc.item())
    loss_ic_history.append(loss_ic.item())

    loss.backward()
    optimizer.step()

    if epoch % (n_epoch//10) == 0:
        elapsed = (perf_counter() - start)/60
        print(f"Time elapsed: {elapsed:.2f} min")
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )

#%%
# Evaluation
n_test = 200

x_test = torch.linspace(-pi, pi, n_test)
y_test = torch.linspace(-pi, pi, n_test)

X_test, Y_test = torch.meshgrid(x_test, y_test, indexing='ij')

t_test = torch.full_like(X_test.reshape(-1,1), 15)

XYT = torch.cat(
    [X_test.reshape(-1,1), Y_test.reshape(-1,1), t_test],
    dim=1
)

net.eval()

with torch.no_grad():
    G_pred = net(XYT)
    
    u_pred = G_pred[:, 0:1]
    v_pred = G_pred[:, 1:2]
    p_pred = G_pred[:, 2:3]

    u_exact = u_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])
    v_exact = v_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])
    p_exact = p_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])

u_pred = u_pred.reshape(n_test, n_test)
v_pred = v_pred.reshape(n_test, n_test)
p_pred = p_pred.reshape(n_test, n_test)

u_exact = u_exact.reshape(n_test, n_test)
v_exact = v_exact.reshape(n_test, n_test)
p_exact = p_exact.reshape(n_test, n_test)

# Convert to numpy for plotting
Xn = X_test.numpy()
Yn = Y_test.numpy()
u_pred_n = u_pred.numpy()
v_pred_n = v_pred.numpy()
p_pred_n = p_pred.numpy()
u_exact_n = u_exact.numpy()
v_exact_n = v_exact.numpy()
p_exact_n = p_exact.numpy()

np.savez(
        f"pinnTaylorGreen_{N}.npz",
        Xn = X_test.numpy(),
        Yn = Y_test.numpy(),
        u_pred_n = u_pred.numpy(),
        v_pred_n = v_pred.numpy(),
        p_pred_n = p_pred.numpy(),
        u_exact_n = u_exact.numpy(),
        v_exact_n = v_exact.numpy(),
        p_exact_n = p_exact.numpy(),
        loss_history = np.array(loss_history),
        loss_pde_history = np.array(loss_pde_history),
        loss_bc_history = np.array(loss_bc_history),
        loss_ic_history = np.array(loss_ic_history)
    )


# # Plot 1
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# # Exact
# cf1 = axes[0].contourf(Xn, Yn, u_exact_n)
# axes[0].set_title("Exact solution")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# fig.colorbar(cf1, ax=axes[0])

# # Prediction
# cf2 = axes[1].contourf(Xn, Yn, u_pred_n)
# axes[1].set_title("PINN prediction")
# axes[1].set_xlabel("x")
# axes[1].set_ylabel("y")
# fig.colorbar(cf2, ax=axes[1])



# # Plot 2
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# # Exact
# cf1 = axes[0].contourf(Xn, Yn, v_exact_n)
# axes[0].set_title("Exact solution")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# fig.colorbar(cf1, ax=axes[0])

# # Prediction
# cf2 = axes[1].contourf(Xn, Yn, v_pred_n)
# axes[1].set_title("PINN prediction")
# axes[1].set_xlabel("x")
# axes[1].set_ylabel("y")
# fig.colorbar(cf2, ax=axes[1])


# # Plot 3
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# # Exact
# cf1 = axes[0].contourf(Xn, Yn, p_exact_n)
# axes[0].set_title("Exact solution")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# fig.colorbar(cf1, ax=axes[0])

# # Prediction
# cf2 = axes[1].contourf(Xn, Yn, p_pred_n)
# axes[1].set_title("PINN prediction")
# axes[1].set_xlabel("x")
# axes[1].set_ylabel("y")
# fig.colorbar(cf2, ax=axes[1])

# #%%
# plt.show()

# plt.figure(figsize=(8, 5))

# plt.semilogy(loss_history, label="Total loss")
# plt.semilogy(loss_pde_history, label="PDE loss")
# plt.semilogy(loss_ic_history, label="IC loss")
# plt.semilogy(loss_bc_history, label="BC loss")

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training loss")
# plt.legend()
# plt.grid(True)
# plt.show()


#%%
