# %%

import numpy as np
import matplotlib.pyplot as plt

data_moe = np.load("moe2dpos_expanded.npz")
data_pinn = np.load("pinn2dpos_expanded.npz")

seeds = np.arange(len(data_moe["L_max"]))

# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(seeds, data_moe["L_max"], label=r"MoE")
ax[0].axhline(np.mean(data_moe["L_max"]), linestyle="--", label=fr"Mean MoE: {np.mean(data_moe['L_max']):.2e}")
ax[0].plot(seeds, data_pinn["L_max"], label=r"PINN", color ="orange")
ax[0].axhline(np.mean(data_pinn["L_max"]), linestyle="--", label=fr"Mean PINN: {np.mean(data_pinn['L_max']):.2e}", color ="orange")
ax[0].set_xlabel("Seed")
ax[0].set_ylabel(r"$L_\infty$ error")
ax[0].set_title(r"$L_\infty$ vs seed")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(seeds, data_moe["L_2"], label=r"MoE")
ax[1].axhline(np.mean(data_moe["L_2"]), linestyle="--", label=fr"Mean MoE: {np.mean(data_moe['L_2']):.2e}")
ax[1].plot(seeds, data_pinn["L_2"], label=r"PINN", color ="orange")
ax[1].axhline(np.mean(data_pinn["L_2"]), linestyle="--", label=fr"Mean PINN: {np.mean(data_pinn['L_2']):.2e}", color ="orange")
ax[1].set_xlabel("Seed")
ax[1].set_ylabel(r"$L_2$ error")
ax[1].set_title(r"$L_2$ vs seed")
ax[1].grid(True)
ax[1].legend()

# %%

plt.tight_layout()
plt.show()

epochs = np.arange(len(data_moe['loss_hist_n']))

plt.figure(figsize=(7, 4))

plt.plot(epochs, data_moe['loss_hist_n'], label="MoE")
plt.plot(epochs, data_pinn['loss_hist_n'], label="PINN", color="orange")

plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.title("Average training loss vs epoch")
plt.yscale("log")   # usually useful for loss curves
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()