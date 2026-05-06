import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("moe2dpos.npz")

Xn = data["Xn"]
Yn = data["Yn"] 
u_pred_n = data["u_pred_n"]
u_exact_n = data["u_exact_n"]
error_n = data["error_n"]

# Use same levels for exact and prediction
vmin = min(u_exact_n.min(), u_pred_n.min())
vmax = max(u_exact_n.max(), u_pred_n.max())
levels = np.linspace(vmin, vmax, 30)

# Separate symmetric scale for error
err_max = np.max(np.abs(error_n))
err_levels = np.linspace(-err_max, err_max, 30)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Exact
cf1 = axes[0].contourf(Xn, Yn, u_exact_n, levels=levels)
axes[0].set_title("Exact solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cf1, ax=axes[0])

# Prediction
cf2 = axes[1].contourf(Xn, Yn, u_pred_n, levels=levels)
axes[1].set_title("MoE prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cf2, ax=axes[1])

# Error
from matplotlib.colors import TwoSlopeNorm

# Force 0 to be the center color
err_norm = TwoSlopeNorm(
    vmin=-err_max,
    vcenter=0.0,
    vmax=err_max
)

cf3 = axes[2].contourf(
    Xn, Yn, error_n,
    levels=30,
    cmap="coolwarm",
    norm=err_norm
)

axes[2].set_title("Error: prediction - exact")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(cf3, ax=axes[2])

plt.show()