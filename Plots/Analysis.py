import numpy as np
import matplotlib.pyplot as plt

base = np.load("base.npz")
true = np.load("true.npz")
softa = np.load("softa.npz")
rar = np.load("rar.npz")
data = np.load("data.npz")
moe_base = np.load("moe_base.npz")

# Plot solution
plt.figure()
plt.plot(base["t"], base["x"], label="base")
plt.plot(true["t"], true["x"], label="solve_ivp")
# plt.plot(softa["t"], softa["x"], "--", label="SoftAdapt")
# plt.plot(rar["t"], rar["x"], "--", label="RAR")
plt.plot(data["t"], data["x"], "--", label="with data")
plt.plot(data["dpt"], data["dpx"], "ro", label="datapoints")
plt.plot(moe_base["t"], moe_base["x"], "--", label="MoE")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Van der Pol oscillator")
plt.legend()
plt.grid()
plt.show()

#%%
# Training loss
# window = 50
# loss_smooth = np.convolve(base["l"], np.ones(window)/window, mode="valid")
# loss_smooth_data = np.convolve(data["l"], np.ones(window)/window, mode="valid")
# plt.figure()
# plt.plot(loss_smooth, label="base")
# plt.plot(loss_smooth_data, label="with data")
# plt.yscale("log")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training loss")
# plt.grid()
# plt.show()

# %%
