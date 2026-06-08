# %%

import numpy as np
import matplotlib.pyplot as plt

# Load data
data_moe = np.load("moe2dpos_expanded.npz")
data_moe_softa = np.load("moe2dpos_expa_softa.npz")
data_pinn = np.load("pinn2dpos_expanded.npz")
data_softa = np.load("pinn2dpos_expa_softa.npz")

err_max1 = np.max(data_moe["error_n"])
err_max2 = np.max(data_moe_softa["error_n"])
err_max3 = np.max(data_pinn["error_n"])
err_max4 = np.max(data_softa["error_n"])

err_max = max(err_max1, err_max2, err_max3, err_max4)

err_min1 = np.min(data_moe["error_n"])
err_min2 = np.min(data_moe_softa["error_n"])
err_min3 = np.min(data_pinn["error_n"])
err_min4 = np.min(data_softa["error_n"])

err_min = min(err_min1, err_min2, err_min3, err_min4)

# %%

for i, result in enumerate(("pinn2dpos_expanded.npz", "moe2dpos_expanded.npz",
                            "moe2dpos_expa_softa.npz","pinn2dpos_expa_softa.npz")):
    data = np.load(result)
    titles = ("PINN", "MoE", "MoE SoftAdapt", "PINN SoftAdapt")

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
    # err_max = np.max(np.abs(error_n))
    err_levels = np.linspace(err_min, err_max, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Exact
    cf1 = axes[0].contourf(Xn, Yn, u_exact_n, levels=levels)
    axes[0].set_title("Exact solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(cf1, ax=axes[0])

    # Prediction
    cf2 = axes[1].contourf(Xn, Yn, u_pred_n, levels=levels)
    axes[1].set_title(f"{titles[i]} average prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(cf2, ax=axes[1])

    # Error
    from matplotlib.colors import TwoSlopeNorm

    # Force 0 to be the center color
    err_norm = TwoSlopeNorm(
        vmin=err_min,
        vcenter=0.0,
        vmax=err_max
    )

    cf3 = axes[2].contourf(
        Xn, Yn, error_n,
        levels=err_levels,
        cmap="coolwarm",
        norm=err_norm
    )

    axes[2].set_title("Error: prediction - exact")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(cf3, ax=axes[2], ticks=np.linspace(err_min, err_max, 7))

    plt.show()

# %%

for i, result in enumerate(("moe2dpos_expanded.npz",
                            "moe2dpos_expa_softa.npz")):
    data = np.load(result)
    titles = ("MoE", "MoE SoftAdapt")


    num_experts = 2

    Xn = data["Xn"]
    Yn = data["Yn"]
    gate_weights = data["gate_weights_n"]

    vmin = np.min(gate_weights)
    vmax = np.max(gate_weights)

    fig, axes = plt.subplots(1, num_experts, figsize=(10, 4), constrained_layout=True)

    for k in range(num_experts):
        cf = axes[k].contourf(
            Xn,
            Yn,
            gate_weights[:, :, k],
            levels=50,
            vmin=vmin,
            vmax=vmax
        )
        axes[k].set_title(f"{titles[i]} Expert {k+1} average weights")
        axes[k].set_xlabel("x")
        axes[k].set_ylabel("y")

    fig.colorbar(cf, ax=axes)
    plt.show()
# %%
