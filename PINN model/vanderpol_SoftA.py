#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from network import PINN
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

torch.manual_seed(0)

#%%
# Parameters
t0 = 0.0
T = 20.0
mu = 5.0

x0 = 2.0
v0 = 0.0

x0_true = torch.tensor([[x0]], dtype=torch.float32)
v0_true = torch.tensor([[v0]], dtype=torch.float32)

#%%
# Initialize network and optimizer
net = PINN(in_dim=1, out_dim=1, hidden_dim=64, hidden_layers=2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#%%
# Numerical reference solution
def f(t, y):
    x, v = y
    a = mu * (1 - x**2) * v - x
    return [v, a]

t_span = (t0, T)
t_eval = np.linspace(t0, T, 10000)
sol = solve_ivp(f, t_span, [x0, v0], t_eval=t_eval)

#%%
# Training setup
softadapt_object = LossWeightedSoftAdapt(beta=0.2)

window = 5
loss_hist_1 = []
loss_hist_2 = []

w_pde = 1.0
w_ic = 10.0

n_epoch = 10_000
N = 800

loss_history = []

for epoch in range(n_epoch):

    optimizer.zero_grad()

    # Collocation points in time
    t = t0 + (T - t0) * torch.rand(N, 1)
    t.requires_grad_(True)

    # Forward pass
    x_hat = net(t)

    # First derivative
    v_hat = torch.autograd.grad(
        x_hat, t,
        grad_outputs=torch.ones_like(x_hat),
        create_graph=True
    )[0]

    # Second derivative
    a_hat = torch.autograd.grad(
        v_hat, t,
        grad_outputs=torch.ones_like(v_hat),
        create_graph=True
    )[0]

    # ODE residual
    residual = a_hat - mu * (1 - x_hat**2) * v_hat + x_hat
    loss_pde = torch.mean(residual**2)

    # Initial conditions at t=0
    t_ic = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True)

    x_ic = net(t_ic)
    v_ic = torch.autograd.grad(
        x_ic, t_ic,
        grad_outputs=torch.ones_like(x_ic),
        create_graph=True
    )[0]

    loss_ic = torch.mean((x_ic - x0_true)**2 + (v_ic - v0_true)**2)

    # save loss components
    loss_hist_1.append(loss_pde.item())
    loss_hist_2.append(loss_ic.item())

    if epoch >= window and epoch % window == 0:
        weights = softadapt_object.get_component_weights(
            torch.tensor(loss_hist_1[-window:], dtype=torch.float32),
            torch.tensor(loss_hist_2[-window:], dtype=torch.float32)
        )
        w_pde, w_ic = [w.item() for w in weights]

    loss = w_pde * loss_pde + w_ic * loss_ic

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(
            f"epoch: {epoch:5d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e} "
            f" w_pde={w_pde:.3f} w_ic={w_ic:.3f}"
            f" w_pde={w_pde:.3f} w_ic={w_ic:.3f}"
        )

#%%
# Evaluation
net.eval()
t_test = torch.linspace(t0, T, 2000).unsqueeze(1)

with torch.no_grad():
    x_pred = net(t_test)

#%%
# Plot solution
plt.figure()
plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(t_test.numpy(), x_pred.numpy(), "--", label="PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Van der Pol oscillator")
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
