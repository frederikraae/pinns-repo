# %%

import numpy as np
import matplotlib.pyplot as plt

# Global fontstørrelse for alle plots
plt.rcParams.update({
    "font.size": 14,          # standard tekst
    "axes.titlesize": 16,     # titler
    "axes.labelsize": 14,     # akse-labels
    "xtick.labelsize": 12,    # x-ticks
    "ytick.labelsize": 12,    # y-ticks
    "legend.fontsize": 10,    # legends
    "figure.titlesize": 18,   # evt. overordnet figur-titel
})

data_moe = np.load("moe2dpos_expanded.npz")
# data_moe_softa = np.load("moe2dpos_expa_softa.npz")
data_pinn = np.load("pinn2dpos_expanded.npz")
data_softa = np.load("pinn2dpos_expa_softa.npz")

seeds = np.arange(len(data_moe["L_max"]))

# %%
titles = ("MoE-PINN", "PINN med SoftAdapt")

for i, model in enumerate((data_moe, data_softa)):

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(seeds, model["L_max"], label=f"{titles[i]}")
    ax[0].axhline(np.mean(model["L_max"]), linestyle="--", label=f"Gennemsnit {titles[i]}: {np.mean(model['L_max']):.2e}")
    ax[0].plot(seeds, data_pinn["L_max"], label=r"PINN", color ="orange")
    ax[0].axhline(np.mean(data_pinn["L_max"]), linestyle="--", label=fr"Gennemsnit PINN: {np.mean(data_pinn['L_max']):.2e}", color ="orange")
    ax[0].set_xlabel("Seed")
    ax[0].set_ylabel(r"$L_\infty$ fejl")
    ax[0].set_title(r"$L_\infty$ vs 'seed'")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(seeds, model["L_2"], label=f"{titles[i]}")
    ax[1].axhline(np.mean(model["L_2"]), linestyle="--", label=f"Gennemsnit {titles[i]}: {np.mean(model['L_2']):.2e}")
    ax[1].plot(seeds, data_pinn["L_2"], label=r"PINN", color ="orange")
    ax[1].axhline(np.mean(data_pinn["L_2"]), linestyle="--", label=fr"Gennemsnit PINN: {np.mean(data_pinn['L_2']):.2e}", color ="orange")
    ax[1].set_xlabel("Seed")
    ax[1].set_ylabel(r"$L_2$ fejl")
    ax[1].set_title(r"$L_2$ vs 'seed'")
    ax[1].grid(True)
    ax[1].legend()

# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(seeds, data_moe["L_max"] - data_pinn["L_max"], label=r"Forskel (MoE-PINN minus PINN)")
ax[0].axhline(np.mean(data_moe["L_max"] - data_pinn["L_max"]),
     linestyle="--", label=fr"Gennemsnitlig forskel: {np.mean(data_moe['L_max'] - data_pinn['L_max']):.2e}")
ax[0].axhline(0, color="red")
ax[0].set_xlabel("Seed")
ax[0].set_ylabel(r"$L_\infty$ fejl forskel")
ax[0].set_title(r"$L_\infty$ forskel vs seed")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(seeds, data_moe["L_2"] - data_pinn["L_2"], label=r"Forskel (MoE-PINN minus PINN)")
ax[1].axhline(np.mean(data_moe["L_2"] - data_pinn["L_2"]),
     linestyle="--", label=fr"Gennemsnitlig forskel: {np.mean(data_moe['L_2'] - data_pinn['L_2']):.2e}")
ax[1].axhline(0, color="red")
ax[1].set_xlabel("Seed")
ax[1].set_ylabel(r"$L_2$ fejl forskel")
ax[1].set_title(r"$L_2$ forskel vs seed")
ax[1].grid(True)
ax[1].legend()

# %%

# Valideringsfejl: L2
epochs = np.arange(len(data_moe["val_l2"]))

plt.figure(figsize=(7, 4))

plt.semilogy(epochs, data_moe["val_l2"], label="MoE-PINN")
plt.semilogy(epochs, data_pinn["val_l2"], label="PINN", color="orange")
# plt.semilogy(epochs, data_moe_softa["val_l2"], label="MoE-PINN med SoftAdapt")
plt.semilogy(epochs, data_softa["val_l2"], label="PINN med SoftAdapt", color ="green")

plt.xlabel("Epoch")
plt.ylabel(r"$L_2$-valideringsfejl")
plt.title(r"Gennemsnitlig $L_2$-valideringsfejl vs 'epoch'")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# %%

# Valideringsfejl: L-infinity
epochs = np.arange(len(data_moe["val_lmax"]))

plt.figure(figsize=(7, 4))

plt.semilogy(epochs, data_moe["val_lmax"], label="MoE-PINN")
plt.semilogy(epochs, data_pinn["val_lmax"], label="PINN", color="orange")
# plt.semilogy(epochs, data_moe_softa["val_lmax"], label="MoE-PINN med SoftAdapt")
plt.semilogy(epochs, data_softa["val_lmax"], label="PINN med SoftAdapt", color ="green")

plt.xlabel("Epoch")
plt.ylabel(r"$L_\infty$-valideringsfejl")
plt.title(r"Gennemsnitlig $L_\infty$-valideringsfejl vs 'epoch'")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Valideringsfejl: L2
epochs = np.arange(len(data_moe["val_l2"]))

ax[0].semilogy(epochs, data_moe["val_lmax"], label="MoE-PINN")
ax[0].semilogy(epochs, data_pinn["val_lmax"], label="PINN", color="orange")
# plt.semilogy(epochs, data_moe_softa["val_lmax"], label="MoE-PINN med SoftAdapt")
ax[0].semilogy(epochs, data_softa["val_lmax"], label="PINN med SoftAdapt", color ="green")

ax[0].set_xlabel("Epoch")
ax[0].set_ylabel(r"$L_\infty$-valideringsfejl")
ax[0].set_title(r"Gennemsnitlig $L_\infty$-valideringsfejl vs 'epoch'")
ax[0].grid(True, which="both")
ax[0].legend()

ax[1].semilogy(epochs, data_moe["val_l2"], label="MoE-PINN")
ax[1].semilogy(epochs, data_pinn["val_l2"], label="PINN", color="orange")
# plt.semilogy(epochs, data_moe_softa["val_l2"], label="MoE-PINN med SoftAdapt")
ax[1].semilogy(epochs, data_softa["val_l2"], label="PINN med SoftAdapt", color ="green")

ax[1].set_xlabel("Epoch")
ax[1].set_ylabel(r"$L_2$-valideringsfejl")
ax[1].set_title(r"Gennemsnitlig $L_2$-valideringsfejl vs 'epoch'")
ax[1].grid(True, which="both")
ax[1].legend()

# %%
