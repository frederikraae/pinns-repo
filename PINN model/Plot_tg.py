#%%

import matplotlib.pyplot as plt
import numpy as np

data = np.load("pinnTaylorGreen.npz")

Xn = data["Xn"]
Yn = data["Yn"]

u_pred_n = data["u_pred_n"]
v_pred_n = data["v_pred_n"]
p_pred_n = data["p_pred_n"]

u_exact_n = data["u_exact_n"]
v_exact_n = data["v_exact_n"]
p_exact_n = data["p_exact_n"]

loss_history = data["loss_history"]
loss_pde_history = data["loss_pde_history"]
loss_bc_history = data["loss_bc_history"]
loss_ic_history = data["loss_ic_history"]

#%%

# Plot 1
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Exact
cf1 = axes[0].contourf(Xn, Yn, u_exact_n)
axes[0].set_title("Exact solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cf1, ax=axes[0])

# Prediction
cf2 = axes[1].contourf(Xn, Yn, u_pred_n)
axes[1].set_title("PINN prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cf2, ax=axes[1])



# Plot 2
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Exact
cf1 = axes[0].contourf(Xn, Yn, v_exact_n)
axes[0].set_title("Exact solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cf1, ax=axes[0])

# Prediction
cf2 = axes[1].contourf(Xn, Yn, v_pred_n)
axes[1].set_title("PINN prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cf2, ax=axes[1])


# Plot 3
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Exact
cf1 = axes[0].contourf(Xn, Yn, p_exact_n)
axes[0].set_title("Exact solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cf1, ax=axes[0])

# Prediction
cf2 = axes[1].contourf(Xn, Yn, p_pred_n)
axes[1].set_title("PINN prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cf2, ax=axes[1])

#%%
plt.show()

plt.figure(figsize=(8, 5))

plt.semilogy(loss_history, label="Total loss")
plt.semilogy(loss_pde_history, label="PDE loss")
plt.semilogy(loss_ic_history, label="IC loss")
plt.semilogy(loss_bc_history, label="BC loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.legend()
plt.grid(True)
plt.show()