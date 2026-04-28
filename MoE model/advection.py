# IMPLEMENTATION OF MOE ARCHITECTURE TO SOLVE THE LINEAR ADVECTION EQUATION
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from new_model import MoEPINN, Expert
from torch import sin, pi

torch.manual_seed(0)

#%%%
# Define wave speed
c = 1.0

# Exact solution
u_exact = lambda x, t: sin(2*pi*(x-c*t))

# Exact initial condition
u_ic_exact = lambda x: sin(2*pi*x)

# Define spation range
xa = 0.0
xb = 1.0

# Define time range
t0 = 0.0
T = 1.0

# Define initial points
N_ic = 100

x_ic = xa + (xb - xa) * torch.rand(N_ic, 1)
t_ic = t0 * torch.ones_like(x_ic)

X_ic = torch.cat([x_ic, t_ic], dim=1)

# Define boundary points
N_bc = 100

t_bc = t0 + (T - t0) * torch.rand(N_bc, 1)
x_bc_a = xa * torch.ones_like(t_bc)
x_bc_b = xb * torch.ones_like(t_bc)

X_bc_a = torch.cat([x_bc_a, t_bc], dim=1)
X_bc_b = torch.cat([x_bc_b, t_bc], dim=1)

# %%
# Initialize MoE model and optimizer
moe = MoEPINN(
    in_dim=2,
    out_dim=1,
    num_experts=3,
    expert_hidden_dim=32,
    expert_hidden_layers=2,
    gate_hidden_dim=16,
    gate_hidden_layers=2,
    temperature=1.0
)

optimizer_moe = torch.optim.Adam(moe.parameters(), lr=1e-4)

#%%
# Initialize PINN model and optimizer
pinn = Expert(
    in_dim=2,
    out_dim=1,
    hidden_dim=64,
    hidden_layers=3
)

optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-4)

#%%
# Training setup
n_epoch = 10000
N = 1000

loss_history_moe = []
loss_history_pinn = []

# Sobol sampling
sobol_moe = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
sobol_pinn = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

#%%
# Training loop for MoE
for epoch in range(n_epoch):
    optimizer_moe.zero_grad()

    # Sample collocation points
    X = sobol_moe.draw(N, dtype=torch.float32)
    X.requires_grad_(True)

    # Forward pass
    u_hat, gate_weights, *_ = moe(X)

    # Compute derivatives
    grad_u_hat = torch.autograd.grad(
        u_hat, X,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    u_x_hat = grad_u_hat[:, 0:1]
    u_t_hat = grad_u_hat[:, 1:2]

    # PDE loss
    residual = u_t_hat + c * u_x_hat
    loss_pde = torch.mean(residual**2)

    # BC loss
    u_bc_a_hat, *_ = moe(X_bc_a)
    u_bc_b_hat, *_ = moe(X_bc_b)
    loss_bc = torch.mean((u_bc_a_hat - u_bc_b_hat)**2)

    # IC loss
    u_ic_hat, *_ = moe(X_ic)
    u_ic_true = u_ic_exact(X_ic)
    loss_ic = torch.mean((u_ic_hat - u_ic_true)**2)

    # Total loss
    loss = loss_pde + loss_bc + loss_ic

    loss_history_moe.append(loss.item())

    loss.backward()
    optimizer_moe.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )

#%%
# Training loop for PINN
for epoch in range(n_epoch):
    optimizer_pinn.zero_grad(True)

    # Sample collocation points
    X = sobol_pinn.draw(N, dtype=torch.float32)
    X.requires_grad_(True)

    # Forward pass
    u = pinn(X)

    # Compute derivatives
    grad_u = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # PDE loss
    residual = u_t + c * u_x
    loss_pde = torch.mean(residual**2)

    # BC loss
    u_bc_a = pinn(X_bc_a)
    u_bc_b = pinn(X_bc_b)
    loss_bc = torch.mean((u_bc_a - u_bc_b)**2)

    # IC loss
    u_ic = pinn(X_ic)
    u_ic_true = u_ic_exact(X_ic)
    loss_ic = torch.mean((u_ic - u_ic_true)**2)

    # Total loss
    loss = loss_pde + loss_bc + loss_ic

    loss_history_pinn.append(loss.item())

    loss.backward()
    optimizer_pinn.step()

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )


#%%
# Evaluation
N_test = 100
moe.eval()
x_test = torch.linspace(xa, xb, N_test)
t_test = torch.linspace(t0, T, N_test)

X_test, T_test = torch.meshgrid(x_test, t_test, indexing='ij')
XT_test = torch.cat([X_test.reshape(-1, 1), T_test.reshape(-1, 1)], dim=1)

with torch.no_grad():
    u_pred_moe, gate_pred, *_ = moe(XT_test)
    u_pred_pinn = pinn(XT_test)
    u_exact_test = u_exact(X_test, T_test)

#%%
# Prepare output for plotting (NumPy)
Xn = X_test.numpy()
Tn = T_test.numpy()
u_pred_moe_n = u_pred_moe.detach().numpy().reshape(Xn.shape)
u_pred_pinn_n = u_pred_pinn.detach().numpy().reshape(Xn.shape)
u_exact_test_n = u_exact_test.numpy()

# Get levels for contour plot
vmin = min(u_exact_test_n.min(), u_pred_moe_n.min())
vmax = max(u_exact_test_n.max(), u_pred_moe_n.max())
levels = np.linspace(vmin, vmax, 30)

#%%
# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12,12), constrained_layout=True)

# Exact
exact_cont = axes[0, 0].contourf(Xn, Tn, u_exact_test_n, levels=levels)
axes[0, 0].set_title("Exact solution")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("t")
fig.colorbar(exact_cont, ax=axes[0,0])

# Prediction MoE
pred_moe_cont = axes[0, 1].contourf(Xn, Tn, u_pred_moe_n, levels=levels)
axes[0, 1].set_title("MoE-PINN predicition")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("t")
fig.colorbar(pred_moe_cont, ax=axes[0,1])

# Prediction PINN
pred_moe_cont = axes[1, 0].contourf(Xn, Tn, u_pred_pinn_n, levels=levels)
axes[1, 0].set_title("PINN predicition")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("t")
fig.colorbar(pred_moe_cont, ax=axes[1,0])

#%%
# Plot training loss for MoE
loss_history_moe = np.array(loss_history_moe)
loss_history_pinn = np.array(loss_history_pinn)
plt.figure()
plt.plot(loss_history_moe, label="MoE-PINN")
plt.plot(loss_history_pinn, color="orange", label="PINN")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid()
plt.show()

#%%
# Visualize gate response
gw = gate_pred.detach().numpy()   # (10000, 3)
levels = np.linspace(0,1,50)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

for k in range(3):
    gw_k = gw[:, k].reshape(Xn.shape)
    cont = axes[k].contourf(Xn, Tn, gw_k, levels=levels)
    axes[k].set_title(f"Gate weight - Expert {k}")
    axes[k].set_xlabel("x")
    axes[k].set_ylabel("t")
    fig.colorbar(cont, ax=axes[k])

plt.show()
#%%