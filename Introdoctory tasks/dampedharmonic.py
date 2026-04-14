#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import scipy as scipy
from network import MLP
from scipy.integrate import solve_ivp

#%%
# Define parameters
t0 = 0.0
t_final = 10.0
lam = 0.5
w0 = 3.0

# Include data point in Loss function
with_data = False

# Define initial conditions
x_t0_true = torch.tensor([[1.0]])
v_t0_true = torch.tensor([[0.0]])

# Initialize network and define optimizer
net = MLP(1,1,[64,64,64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

w_ic = 2.0 # weight for i.c. loss

# %%

# Numeric integration of system
def f(t, y):
    x, v = y
    a = -w0**2 * x - lam * v
    return [v, a]

# Initial conditions (convert to float)
x0_true = x_t0_true.squeeze().item()
v0_true = v_t0_true.squeeze().item()
y0 = [x0_true, v0_true]   # x(0), v(0)

# Time interval
t_intervals = 10_000
t_span = (t0, t_final)
t_eval = np.linspace(t0, t_final, t_intervals)

# Solve
sol = solve_ivp(f, t_span, y0, t_eval=t_eval)

#%%

# Define datapoint
tdata = 4.0 # in time domain

t_index = np.argmin(np.abs(t_eval - tdata)) # index for solution corresponing to tdata
xdata = sol.y[0][t_index]

tdata_tensor = torch.tensor([[tdata]], dtype=torch.float32)
x_data_true = torch.tensor([[xdata]], dtype=torch.float32)

w_data = 5.0 # weight for data loss

#%%
# Training loop
n_epoch = 20_000

if with_data:
    loss_history_data = []
else:
    loss_history = []

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

    # Datapoint loss
    if with_data:
        x_data = net(tdata_tensor)

        loss_data = torch.mean((x_data - x_data_true)**2)
    else:
        loss_data = torch.tensor(0.0, dtype=torch.float32)

    # Total loss
    loss = loss_pde + w_ic * loss_ic + w_data * loss_data

    loss.backward()
    optimizer.step()

    if with_data:
        loss_history_data.append(loss.item())
    else:
        loss_history.append(loss.item())

    if epoch % (n_epoch//10) == 0:
        print(
            f"epoch: {epoch:4d} "
            f"loss={loss.item():.6e} "
            f"pde={loss_pde.item():.6e} "
            f"ic={loss_ic.item():.6e}"
        )

#%%
# Evaluation
net.eval()
t_test = torch.linspace(t0, t_final, 10000).unsqueeze(1)

with torch.no_grad():
    x_pred = net(t_test)
plt.figure()
# Plot PINN approximation
plt.plot(t_test.numpy(), x_pred.numpy(), label="PINN")

# Plot numeric solution
plt.plot(sol.t, sol.y[0], label="Numeric solution")

# Datapoint
if with_data:
    plt.scatter(tdata, xdata, color="red", label="Datapoint", zorder=5)

plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.title("Damped Harmonic Oscillator - PINN")
plt.legend()
plt.show()

#%%
# Training loss

loss_history_data = np.array(loss_history_data)
loss_history = np.array(loss_history)

# Moving average
window = 30
loss_history_data = np.convolve(loss_history_data, np.ones(window)/window, mode='valid')
loss_history = np.convolve(loss_history, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(loss_history, label = "No datapoint")
plt.plot(loss_history_data, label = "With datapoint")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid()
plt.title("Training loss")
plt.legend()
plt.show()

print(f'Epochs: {n_epoch}')
# %%
