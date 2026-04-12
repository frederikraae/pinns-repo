import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuralop.models import FNO

torch.manual_seed(0)

# ----------------------------
# Make dataset
# ----------------------------
n_samples = 200   # number of functions
n_grid = 128      # spatial resolution
n_modes = 8       # number of sine modes
L = 10            # domain length

x = torch.linspace(0, L, n_grid)

f_data = torch.zeros(n_samples, n_grid)
u_data = torch.zeros(n_samples, n_grid)

for k in range(1, n_modes + 1):
    a = torch.randn(n_samples, 1)
    basis = torch.sin(k * np.pi * x / L).view(1, n_grid)
    f_data += a * basis
    u_data += a / ((k * np.pi / L) ** 2) * basis

f_data = f_data.unsqueeze(1)
u_data = u_data.unsqueeze(1)

x_channel = x.view(1, 1, n_grid).repeat(n_samples, 1, 1)

X = torch.cat([f_data, x_channel], dim=1)
Y = u_data

# ----------------------------
# Normalize
# ----------------------------
y_mean = Y.mean()
y_std = Y.std()

Y_norm = (Y - y_mean) / y_std

# ----------------------------
# Define FNO
# ----------------------------
model = FNO(
    n_modes=(16,),
    hidden_channels=32,
    in_channels=2,
    out_channels=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ----------------------------
# Train (data phase)
# ----------------------------
n_epochs = 100

for epoch in range(n_epochs):
    pred = model(X)
    loss = loss_fn(pred, Y_norm)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Data] Epoch {epoch+1:3d}, loss = {loss.item():.6e}")

# ----------------------------
# Physics helpers
# ----------------------------
dx = x[1] - x[0]

def second_derivative(u, dx):
    u_left  = u[:, :, :-2]
    u_mid   = u[:, :, 1:-1]
    u_right = u[:, :, 2:]
    return (u_left - 2*u_mid + u_right) / dx**2

def physics_loss(u_pred, f, dx):
    u_xx = second_derivative(u_pred, dx)
    f_interior = f[:, :, 1:-1]
    residual = -u_xx - f_interior
    return (residual**2).mean()

def boundary_loss(u_pred):
    left = u_pred[:, :, 0]
    right = u_pred[:, :, -1]
    return (left**2 + right**2).mean()

# ----------------------------
# Physics fine-tuning
# ----------------------------
n_phys_epochs = 100
lambda_phys = 0.001
lambda_bc = 1.0

for epoch in range(n_phys_epochs):
    pred_norm = model(X)
    pred = pred_norm * y_std + y_mean

    loss_data = loss_fn(pred_norm, Y_norm)
    loss_phys = physics_loss(pred, f_data, dx)
    loss_bc = boundary_loss(pred)

    loss = loss_data + lambda_phys * loss_phys + lambda_bc * loss_bc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Hybrid] Epoch {epoch+1}, loss = {loss.item():.6e}")

# ----------------------------
# Plot
# ----------------------------
with torch.no_grad():
    i = 0
    pred_norm = model(X[i:i+1])
    pred = (pred_norm * y_std + y_mean).squeeze()

plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), f_data[i, 0].numpy(), "--", label="f(x)")
plt.plot(x.numpy(), Y[i, 0].numpy(), label="true u(x)")
plt.plot(x.numpy(), pred.numpy(), label="predicted u(x)")
plt.xlabel("x")
plt.legend()
plt.title("1D Poisson with FNO + Physics Fine-Tuning")
plt.tight_layout()
plt.show()