#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from new_model import MoEPINN
from torch import sin, exp, pi

torch.manual_seed(0)

#%%
# Parameters for u
a = 5
mu = 0.3
c = 20

# Boundary points
a = 0.0
b = 1.0

# Define chosen u and forcing term
u_sharp = lambda x: exp(-500*(x-0.3)**2)
u_smooth = lambda x: 2 * sin(pi*x)
u_chos = lambda x: u_sharp(x) + u_smooth(x)

f = lambda x: -(1000*x - 300)**2 * exp(-500*(x-0.3)**2) + 2*pi**2 * sin(pi*x) + 1000*exp(-500*(x-0.3)**2)

#%%
# Plot chosen u(x) and f(x)
plot_u_f = 0
x = torch.linspace(a,b,1000)

if plot_u_f:
    fig, ax1 = plt.subplots()

    line1, = ax1.plot(x, u_chos(x), color='teal', label="u(x)")
    ax1.set_ylabel("u(x)")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    line2, = ax2.plot(x, f(x), '--', color='orange', label="-f(x)")
    ax2.set_ylabel("f(x)")
    ax2.tick_params(axis='y')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    plt.show()
#%%
# Initialize MoE and define optimizer
moe = MoEPINN(
    in_dim=1,
    out_dim=1,
    num_experts=2,
    expert_hidden_dim=32,
    expert_hidden_layers=2,
    gate_hidden_dim=16,
    gate_hidden_layers=2,
    temperature=5.0
)

optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

# %%
# Training setup
n_epoch = 10000
N = 1000

loss_history = []

# Sobol sampling
sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Sample collocation points
    x = sobol.draw(N, dtype=torch.float32)
    x.requires_grad_(True)

    # Forward pass
    u_hat, gate_weights, _, _ = moe(x)

    # Get derivatives
    up_hat = torch.autograd.grad(
        u_hat, x,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    upp_hat = torch.autograd.grad(
        up_hat, x,
        grad_outputs=torch.ones_like(up_hat),
        create_graph=True
    )[0]

    # PDE residual
    residual = upp_hat + f(x)
    loss_pde = torch.mean(residual**2)

    # BC loss
    x_a = torch.tensor([[a]])
    x_b = torch.tensor([[b]])

    u_a, _, _, _ = moe(x_a)
    u_b, _, _, _ = moe(x_b)

    loss_bc = (u_a - u_chos(x_a))**2 + (u_b - u_chos(x_b))**2
    loss_bc = loss_bc.mean()

    # Load balance (gate loss)
    mean_gate = torch.mean(gate_weights.squeeze(1), dim=0)
    K = moe.num_experts
    loss_balance = K * torch.sum(mean_gate**2)


    # Total loss
    loss = loss_pde + loss_bc + loss_balance

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item() - loss_balance.item())

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"bc={loss_bc.item():.6e} "
            f"bal={loss_balance.item():.6e}"
        )

#%%
# Evaluation
N_test = 2000
moe.eval()
x_test = torch.linspace(a,b,N_test).unsqueeze(1)

with torch.no_grad():
    u_pred, gate_pred, _, _ = moe(x_test)

# Plot solution
plt.figure()
plt.plot(x_test, u_chos(x_test), label="Exact solution")
plt.plot(x_test.numpy(), u_pred.numpy(), "--", label="MoE-PINN")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Poisson equation with heterogeneous source")
plt.legend()
plt.grid()
plt.show()

#%%
# Plot gate responses
plt.figure()
for k in range(moe.num_experts):
    plt.plot(x_test.numpy(), gate_pred[:, k].numpy(), label=f"Expert {k+1}")
plt.xlabel("x")
plt.ylabel("Gate weight")
plt.title("Gate responses over time")
plt.legend()
plt.grid()
plt.show()

#%%
# Training loss
loss_history = np.array(loss_history)
window = 50
loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode="valid")
plt.figure()
plt.plot(loss_smooth)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid()
plt.show()

#%%


