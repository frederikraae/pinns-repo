#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import scipy as scipy
from network import MLP, MoEPINN
from scipy.integrate import solve_ivp

torch.manual_seed(0)

#%%

# Problem setup
t0 = 0.0
T = 2.0
u0 = 1.0

# Initialize network and define optimizer
moe_model = MoEPINN(input_dim=2, output_dim=1, hidden=[32, 32], n_experts=2)
optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3)

# %%

# Numeric integration of system
def f(t, u, lam):
    return lam * u

t_span = (t0, T)
t_eval = np.linspace(t0, T, 10000)

sol_neg = solve_ivp(lambda t, u: f(t, u, -1.0), t_span, [u0], t_eval=t_eval)
sol_pos = solve_ivp(lambda t, u: f(t, u,  1.0), t_span, [u0], t_eval=t_eval)

#%%
# Training loop
n_epoch = 20_000
N = 1000
w_ic = 10.0
w_balance = 1e-2

lam_neg = -torch.ones(N, 1)
lam_pos =  torch.ones(N, 1)

loss_history = []

for epoch in range(n_epoch):
    optimizer.zero_grad()

    t = t0 + (T - t0) * torch.rand(N, 1)

    lam_neg = -torch.ones(N, 1)
    lam_pos =  torch.ones(N, 1)

    x_neg = torch.cat([t, lam_neg], dim=1)
    x_pos = torch.cat([t, lam_pos], dim=1)
    x.requires_grad_(True)

    x = torch.cat([x_neg, x_pos], dim=0)

    # Forward pass
    u_hat, gate_weights = moe_model(x)

    # Extract t and lambda
    t = x[:, 0:1]
    lam = x[:, 1:2]

    # du/dt
    du_dt = torch.autograd.grad(
        u_hat, x,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0][:, 0:1]

    # PDE residual: u_t - lambda*u = 0
    residual = du_dt - lam * u_hat
    loss_pde = torch.mean(residual**2)

    # Initial condition loss for both lambda = -1 and +1
    x_ic = torch.tensor([
        [t0, -1.0],
        [t0,  1.0]
    ], dtype=torch.float32, requires_grad=True)

    u_ic, gate_ic = moe_model(x_ic)
    u0_true = u0 * torch.ones_like(u_ic)
    loss_ic = torch.mean((u_ic - u0_true)**2)

    # Optional load-balancing loss
    mean_gate = torch.mean(gate_weights.squeeze(1), dim=0)
    target = torch.full_like(mean_gate, 1.0 / moe_model.n_experts)
    loss_balance = torch.mean((mean_gate - target)**2)

    # Total loss
    loss = loss_pde + w_ic * loss_ic + w_balance * loss_balance

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 2000 == 0:
        print(
            f"epoch: {epoch:5d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e} "
            f"bal={loss_balance.item():.6e}"
        )

#%%
moe_model.eval()

t_test = torch.linspace(t0, T, 1000).unsqueeze(1)

x_test_neg = torch.cat([t_test, -torch.ones_like(t_test)], dim=1)
x_test_pos = torch.cat([t_test,  torch.ones_like(t_test)], dim=1)

with torch.no_grad():
    u_pred_neg, gate_neg = moe_model(x_test_neg)
    u_pred_pos, gate_pos = moe_model(x_test_pos)

plt.figure()
plt.plot(sol_neg.t, sol_neg.y[0], label="solve_ivp, λ=-1")
plt.plot(t_test.numpy(), u_pred_neg.numpy(), "--", label="MoE-PINN, λ=-1")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(sol_pos.t, sol_pos.y[0], label="solve_ivp, λ=1")
plt.plot(t_test.numpy(), u_pred_pos.numpy(), "--", label="MoE-PINN, λ=1")
plt.legend()
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(t_test.numpy(), gate_neg[:, 0, 0].numpy(), label="Expert 1, λ=-1")
plt.plot(t_test.numpy(), gate_neg[:, 0, 1].numpy(), label="Expert 2, λ=-1")
plt.plot(t_test.numpy(), gate_pos[:, 0, 0].numpy(), "--", label="Expert 1, λ=1")
plt.plot(t_test.numpy(), gate_pos[:, 0, 1].numpy(), "--", label="Expert 2, λ=1")
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel("Gate weight")
plt.title("Gate responses")
plt.show()

#%%
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
du_dt = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0][:, 0:1]