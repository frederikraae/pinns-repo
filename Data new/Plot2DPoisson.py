# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Fælles fontstørrelser for alle plots
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
CBAR_TICK_SIZE = 12

plt.rcParams.update({
    "font.size": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": 12,
})

# Indlæs data
data_moe = np.load("moe2dpos_expanded.npz")
# data_moe_softa = np.load("moe2dpos_expa_softa.npz")
data_pinn = np.load("pinn2dpos_expanded.npz")
data_softa = np.load("pinn2dpos_expa_softa.npz")

# Find fælles farveskala for fejlen på tværs af alle modeller
err_max1 = np.max(data_moe["error_n"])
# err_max2 = np.max(data_moe_softa["error_n"])
err_max3 = np.max(data_pinn["error_n"])
err_max4 = np.max(data_softa["error_n"])

err_max = max(err_max1, err_max3, err_max4)

err_min1 = np.min(data_moe["error_n"])
# err_min2 = np.min(data_moe_softa["error_n"])
err_min3 = np.min(data_pinn["error_n"])
err_min4 = np.min(data_softa["error_n"])

err_min = min(err_min1, err_min3, err_min4)

# %%

# Konturplots af eksakt løsning, gennemsnitlig prædiktion og fejl
for i, result in enumerate((
    "pinn2dpos_expanded.npz",
    "moe2dpos_expanded.npz",
    "pinn2dpos_expa_softa.npz"
)):
    data = np.load(result)

    titles = (
        "PINN",
        "MoE-PINN",
        "PINN med SoftAdapt"
    )

    Xn = data["Xn"]
    Yn = data["Yn"]
    u_pred_n = data["u_pred_n"]
    u_exact_n = data["u_exact_n"]
    error_n = data["error_n"]

    # Brug samme niveauer for eksakt løsning og prædiktion
    vmin = min(u_exact_n.min(), u_pred_n.min())
    vmax = max(u_exact_n.max(), u_pred_n.max())
    levels = np.linspace(vmin, vmax, 30)

    # Fælles farveskala for fejlen på tværs af alle modeller
    err_levels = np.linspace(err_min, err_max, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Eksakt løsning
    cf1 = axes[0].contourf(Xn, Yn, u_exact_n, levels=levels)
    axes[0].set_title("Eksakt løsning")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    cbar1 = fig.colorbar(cf1, ax=axes[0])
    cbar1.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # Gennemsnitlig prædiktion
    cf2 = axes[1].contourf(Xn, Yn, u_pred_n, levels=levels)
    axes[1].set_title(f"{titles[i]}:\n gennemsnitlig prædiktion")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    cbar2 = fig.colorbar(cf2, ax=axes[1])
    cbar2.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # Fejl
    err_norm = TwoSlopeNorm(
        vmin=err_min,
        vcenter=0.0,
        vmax=err_max
    )

    cf3 = axes[2].contourf(
        Xn,
        Yn,
        error_n,
        levels=err_levels,
        cmap="coolwarm",
        norm=err_norm
    )

    axes[2].set_title("Fejl:\n prædiktion minus eksakt løsning")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$y$")
    cbar3 = fig.colorbar(cf3, ax=axes[2], ticks=np.linspace(err_min, err_max, 7))
    cbar3.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    plt.show()

# %%

# Konturplots af gennemsnitlige gate-vægte for MoE-modellerne
for i, result in enumerate((
    "moe2dpos_expanded.npz",
    "moe2dpos_expa_softa.npz"
)):
    data = np.load(result)

    titles = (
        "MoE-PINN",
        "MoE-PINN med SoftAdapt"
    )

    num_experts = 2

    Xn = data["Xn"]
    Yn = data["Yn"]
    gate_weights = data["gate_weights_n"]

    vmin = np.min(gate_weights)
    vmax = np.max(gate_weights)

    fig, axes = plt.subplots(
        1,
        num_experts,
        figsize=(10, 4),
        constrained_layout=True
    )

    for k in range(num_experts):
        cf = axes[k].contourf(
            Xn,
            Yn,
            gate_weights[:, :, k],
            levels=50,
            vmin=vmin,
            vmax=vmax
        )

        axes[k].set_title(
            f"{titles[i]}: \n gennemsnitlig vægt for ekspert {k+1}"
        )
        axes[k].set_xlabel("$x$")
        axes[k].set_ylabel("$y$")

    fig.colorbar(cf, ax=axes)
    plt.show()

# %%