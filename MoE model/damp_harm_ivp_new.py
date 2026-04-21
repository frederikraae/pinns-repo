#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from new_model import MoEPINN

torch.manual_seed(0)

#%%
# Parameters
t0 = 0.0
T = 20.0
lam = 0.5
w0 = 3.0

x0 = 1.0
v0 = 0.0

x0_true = torch.tensor([[x0]], dtype=torch.float32)
v0_true = torch.tensor([[v0]], dtype=torch.float32)

#%%
# MoE model: input is only t
moe_model = MoEPINN(in_dim=1, out_dim=1, num_experts=3, temperature=1)
optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3)

#%%
# Numerical reference solution
def f(t, y):
    x, v = y
    a = -lam * v - w0**2 * x
    return [v, a]

t_span = (t0, T)
t_eval = np.linspace(t0, T, 10000)
sol = solve_ivp(f, t_span, [x0, v0], t_eval=t_eval)

#%%
# Training setup
n_epoch = 20000
N = 200

loss_history = []

#%%
for epoch in range(n_epoch):

    optimizer.zero_grad()

    # collocation points in time
    t = t0 + (T - t0) * torch.rand(N, 1)
    t.requires_grad_(True)

    # forward
    x_hat, gate_weights, _, _ = moe_model(t)

    # first derivative
    v_hat = torch.autograd.grad(
        x_hat, t,
        grad_outputs=torch.ones_like(x_hat),
        create_graph=True
    )[0]

    # second derivative
    a_hat = torch.autograd.grad(
        v_hat, t,
        grad_outputs=torch.ones_like(v_hat),
        create_graph=True
    )[0]

    # ODE residual: x'' + lam x' + w0^2 x = 0
    residual = a_hat + lam * v_hat + w0**2 * x_hat
    loss_pde = torch.mean(residual**2)

    # initial conditions at t=0
    t_ic = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True)

    x_ic, _, _, _ = moe_model(t_ic)
    v_ic = torch.autograd.grad(
        x_ic, t_ic,
        grad_outputs=torch.ones_like(x_ic),
        create_graph=True
    )[0]

    loss_ic = torch.mean((x_ic - x0_true)**2 + (v_ic - v0_true)**2)

    # Load balancing consistent with project plan
    # mean_gate = torch.mean(gate_weights.squeeze(1), dim=0) 
    # K = moe_model.n_experts
    # loss_balance = K * torch.sum(mean_gate ** 2)

    w_pde = 1.0
    w_ic = 1.0
    w_balance = 0.0

    loss = w_pde * loss_pde + w_ic * loss_ic # + w_balance * loss_balance

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        print(
            f"epoch: {epoch:5d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e} "
            #f"bal={loss_balance.item():.6e}"
            f" w_pde={w_pde:.3f} w_ic={w_ic:.3f} w_bal={w_balance:.3f}"
        )

#%%
# Evaluation
moe_model.eval()
t_test = torch.linspace(t0, T, 2000).unsqueeze(1)

with torch.no_grad():
    x_pred, gate_pred, _, _ = moe_model(t_test)

#%%
# Plot solution
plt.figure()
plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(t_test.numpy(), x_pred.numpy(), "--", label="MoE-PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Damped harmonic oscillator")
plt.legend()
plt.grid()
plt.show()

#%%
# Plot gate responses
plt.figure()
for k in range(moe_model.num_experts):
    plt.plot(t_test.numpy(), gate_pred[:, k].numpy(), label=f"Expert {k+1}")
plt.xlabel("t")
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
# %%
