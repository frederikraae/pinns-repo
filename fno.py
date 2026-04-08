import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuralop.models import FNO

torch.manual_seed(0)

# Make dataset
n_samples = 200
n_grid = 128
n_modes = 8
L = 10

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

# Y statistics
y_mean = Y.mean()
y_std = Y.std()

# Normalize Y
Y_norm = (Y - y_mean) / y_std

# Define FNO
model = FNO(
    n_modes=(16,),
    hidden_channels=32,
    in_channels=2,
    out_channels=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train
n_epochs = 100

for epoch in range(n_epochs):
    pred = model(X)
    loss = loss_fn(pred, Y_norm)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1:2d}, loss = {loss.item():.6e}")

# Plot
with torch.no_grad():
    i = 0
    pred_norm = model(X[i:i+1]).squeeze()
    pred = (pred_norm * y_std + y_mean).squeeze()

plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), f_data[i, 0].numpy(), "--", label="f(x)")
plt.plot(x.numpy(), Y[i, 0].numpy(), label="true u(x)")
plt.plot(x.numpy(), pred.numpy(), label="predicted u(x)")
plt.xlabel("x")
plt.legend()
plt.title("1D Poisson with a tiny FNO")
plt.tight_layout()
plt.show()