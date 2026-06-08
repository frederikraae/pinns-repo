# %%

import numpy as np
import matplotlib.pyplot as plt

data_moe = np.load("moe2dpos_expanded.npz")
data_moe_softa = np.load("moe2dpos_expa_softa.npz")
data_pinn = np.load("pinn2dpos_expanded.npz")
data_softa = np.load("pinn2dpos_expa_softa.npz")

seeds = np.arange(len(data_moe["L_max"]))

# %%
titles = ("MoE","MoE SoftAdapt", "PINN SoftAdapt")

for i, model in enumerate((data_moe, data_moe_softa, data_softa)):

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(seeds, model["L_max"], label=f"{titles[i]}")
    ax[0].axhline(np.mean(model["L_max"]), linestyle="--", label=f"Mean {titles[i]}: {np.mean(model['L_max']):.2e}")
    ax[0].plot(seeds, data_pinn["L_max"], label=r"PINN", color ="orange")
    ax[0].axhline(np.mean(data_pinn["L_max"]), linestyle="--", label=fr"Mean PINN: {np.mean(data_pinn['L_max']):.2e}", color ="orange")
    ax[0].set_xlabel("Seed")
    ax[0].set_ylabel(r"$L_\infty$ error")
    ax[0].set_title(r"$L_\infty$ vs seed")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(seeds, model["L_2"], label=f"{titles[i]}")
    ax[1].axhline(np.mean(model["L_2"]), linestyle="--", label=f"Mean {titles[i]}: {np.mean(model['L_2']):.2e}")
    ax[1].plot(seeds, data_pinn["L_2"], label=r"PINN", color ="orange")
    ax[1].axhline(np.mean(data_pinn["L_2"]), linestyle="--", label=fr"Mean PINN: {np.mean(data_pinn['L_2']):.2e}", color ="orange")
    ax[1].set_xlabel("Seed")
    ax[1].set_ylabel(r"$L_2$ error")
    ax[1].set_title(r"$L_2$ vs seed")
    ax[1].grid(True)
    ax[1].legend()

# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(seeds, data_moe["L_max"] - data_pinn["L_max"], label=r"Difference (MoE-PINN)")
ax[0].axhline(np.mean(data_moe["L_max"] - data_pinn["L_max"]),
     linestyle="--", label=fr"Mean difference: {np.mean(data_moe['L_max'] - data_pinn['L_max']):.2e}")
ax[0].axhline(0, color="red")
ax[0].set_xlabel("Seed")
ax[0].set_ylabel(r"$L_\infty$ error difference")
ax[0].set_title(r"$L_\infty$ difference vs seed")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(seeds, data_moe["L_2"] - data_pinn["L_2"], label=r"Difference (MoE-PINN)")
ax[1].axhline(np.mean(data_moe["L_2"] - data_pinn["L_2"]),
     linestyle="--", label=fr"Mean difference: {np.mean(data_moe['L_2'] - data_pinn['L_2']):.2e}")
ax[1].axhline(0, color="red")
ax[1].set_xlabel("Seed")
ax[1].set_ylabel(r"$L_2$ error difference")
ax[1].set_title(r"$L_2$ difference vs seed")
ax[1].grid(True)
ax[1].legend()

# %%

plt.tight_layout()
plt.show()

epochs = np.arange(len(data_moe['loss_hist_n']))

plt.figure(figsize=(7, 4))

plt.plot(epochs, data_moe['loss_hist_n'], label="MoE")
plt.plot(epochs, data_pinn['loss_hist_n'], label="PINN", color="orange")
plt.plot(epochs, data_moe_softa['loss_hist_n'], label="MoE SoftAdapt")
plt.plot(epochs, data_softa['loss_hist_n'], label="PINN SoftAdapt")

plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.title("Average training loss vs epoch")
plt.yscale("log")   # usually useful for loss curves
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
