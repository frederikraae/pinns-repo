#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from new_model import MoEPINN, Expert
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

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
# Numerical reference solution
def f(t, y):
    x, v = y
    a = -lam * v - w0**2 * x
    return [v, a]

t_span = (t0, T)
t_eval = np.linspace(t0, T, 10000)
sol = solve_ivp(f, t_span, [x0, v0], t_eval=t_eval)

#%%
# Initialize MoE model and define MoE-optimizer
moe = moe = MoEPINN(
    in_dim=1,
    out_dim=1,
    num_experts=2,
    expert_hidden_dim=32,
    expert_hidden_layers=2,
    gate_hidden_dim=16,
    gate_hidden_layers=2,
    temperature=5.0
)
optimizer_moe = torch.optim.Adam(moe.parameters(), lr=1e-3)

#%%
# Initialize PINN and define PINN-optimizer
pinn = Expert(
    in_dim=11,
    out_dim=1,
    hidden_dim=32,
    hidden_layers=2
)

optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-3)

#%%
# Define data transformer
def datatransform(t, l_fun=[torch.cos, torch.sin], l_freq=[1.0, 2.0, 4.0, 8.0, 16.0]):
    t_trans = t
    for fun in l_fun:
        for freq in l_freq:
            t_trans = torch.cat([t_trans, fun(2*torch.pi*freq*t)], dim=1)
    return t_trans

#%%
# Training setup
softadapt_object  = LossWeightedSoftAdapt(beta=0.2)

n_epoch = 10_000
N = 800

loss_history = []

window = 5

loss_hist_1 = []
loss_hist_2 = []
loss_hist_3 = []

# initial weights before SoftAdapt can be computed
w_pde, w_ic, w_balance = 1.0, 1.0, 1.0

#%%
# Training loop for MoE
for epoch in range(n_epoch):

    optimizer_moe.zero_grad()

    # collocation points in time
    t = t0 + (T - t0) * torch.rand(N, 1)
    t.requires_grad_(True)

    # forward
    x_hat, gate_weights, _, _ = moe(t)

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

    x_ic, _, _, _ = moe(t_ic)
    v_ic = torch.autograd.grad(
        x_ic, t_ic,
        grad_outputs=torch.ones_like(x_ic),
        create_graph=True
    )[0]

    loss_ic = torch.mean((x_ic - x0_true)**2 + (v_ic - v0_true)**2)

    # Load balancing consistent with project plan
    mean_gate = torch.mean(gate_weights.squeeze(1), dim=0) 
    K = moe_model.num_experts
    loss_balance = (1 - K * torch.sum(mean_gate ** 2))**2

    # save loss components
    loss_hist_1.append(loss_pde.item())
    loss_hist_2.append(loss_ic.item())
    loss_hist_3.append(loss_balance.item())

    if epoch >= window and epoch % window == 0:
        weights = softadapt_object.get_component_weights(
            torch.tensor(loss_hist_1[-window:], dtype=torch.float32),
            torch.tensor(loss_hist_2[-window:], dtype=torch.float32),
            torch.tensor(loss_hist_3[-window:], dtype=torch.float32)
        )
        w_pde, w_ic, w_balance = [w.item() for w in weights]

    loss = w_pde * loss_pde + w_ic * loss_ic # + w_balance * loss_balance

    loss.backward()
    optimizer_moe.step()

    loss_history_moe.append(loss.item())

    if epoch % 200 == 0:
        print(
            f"epoch: {epoch:5d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e} "
            f"bal={loss_balance.item():.6e}"
            f" w_pde={w_pde:.3f} w_ic={w_ic:.3f} w_bal={w_balance:.3f}"
        )
#%%
# Training loop for PINN
for epoch in range(n_epoch):
    optimizer_pinn.zero_grad(True)

    # Sample collocation points
    t = t0 + (T - t0) * torch.rand(N, 1)
    t.requires_grad_(True)
    t_enc = datatransform(t)

    # Forward pass
    x_hat = pinn(t_enc)

    # Get derivatives
    v_hat = torch.autograd.grad(
        x_hat, t,
        grad_outputs=torch.ones_like(x_hat),
        create_graph=True
    )[0]

    a_hat = torch.autograd.grad(
        v_hat, t,
        grad_outputs=torch.ones_like(v_hat),
        create_graph=True
    )[0]

    # ODE residual
    residual = a_hat + lam * v_hat + w0**2 * x_hat
    loss_pde = torch.mean(residual**2)

    # IC loss
    t_ic = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True)
    t_ic_enc = datatransform(t_ic)

    x_ic = pinn(t_ic_enc)
    v_ic = torch.autograd.grad(
        x_ic, t_ic,
        grad_outputs=torch.ones_like(x_ic),
        create_graph=True
    )[0]

    loss_ic = torch.mean((x_ic - x0_true)**2 + (v_ic - v0_true)**2)

    w_pde = 100.0
    w_ic = 1000.0

    loss = w_pde * loss_pde + w_ic * loss_ic

    loss.backward()
    optimizer_pinn.step()

    loss_history_pinn.append(loss.item())

    if epoch % 1000 == 0:
        print(
            f"epoch: {epoch:5d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e}")



#%%
# Evaluation
N_test = 2000
moe.eval()
pinn.eval()
t_test1 = torch.linspace(t0, T, N_test).unsqueeze(1)
t_test2 = datatransform(t_test1)

with torch.no_grad():
    x_pred_moe, gate_pred, _, _ = moe(t_test1)
    x_pred_pinn = pinn(t_test2)

#%%
# Plot solution
plt.figure()
plt.plot(sol.t, sol.y[0], label="solve_ivp")
# plt.plot(t_test1.numpy(), x_pred_moe.numpy(), "--", label="MoE-PINN")
plt.plot(t_test1.numpy(), x_pred_pinn.numpy(), "--", color="green", label="PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Damped harmonic oscillator")
plt.legend()
plt.grid()
plt.show()

#%%
# Training loss
loss_history_moe = np.array(loss_history_moe)
loss_history_pinn = np.array(loss_history_pinn)
plt.figure()
plt.plot(loss_history_moe, color="orange", label="MoE-PINN")
plt.plot(loss_history_pinn, color="green", label="PINN")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.legend()
plt.grid()
plt.show()

#%%
# Plot gate responses
plt.figure()
for k in range(moe.num_experts):
    plt.plot(t_test1.numpy(), gate_pred[:, k].numpy(), label=f"Expert {k+1}")
plt.xlabel("t")
plt.ylabel("Gate weight")
plt.title("Gate responses over time")
plt.legend()
plt.grid()
plt.show()

# %%

