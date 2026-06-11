# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Global font sizes
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
})

# Baseline file
baseline_file = "pinn2dpos_expanded.npz"

# Files to compare against baseline
files = [
    "moe2dpos_expanded.npz",
    "pinn2dpos_expa_softa.npz",
]

model_colors = {
    baseline_file: "tab:blue",
    "moe2dpos_expanded.npz": "tab:orange",
    "pinn2dpos_expa_softa.npz": "tab:green",
}

plot_name = {
    baseline_file: "PINN",
    "moe2dpos_expanded.npz": "MoE-PINN",
    "pinn2dpos_expa_softa.npz": "PINN med SoftAdapt",
}

N_LEVELS = 30

# Load datasets
baseline_data = np.load(baseline_file)
datasets = [np.load(file) for file in files]

#%%

# ------------------------------------------------------------
# Find common color limits for contour plots
# ------------------------------------------------------------

all_datasets = [baseline_data] + datasets

u_min, u_max = np.inf, -np.inf
e_min, e_max = 0.0, 0.0

for data in all_datasets:
    u_pred_n = data["u_pred_n"]

    u_exact_n = data["u_exact_n"]

    u_min = min(u_min, u_pred_n.min(), u_exact_n.min())
    u_max = max(u_max, u_pred_n.max(), u_exact_n.max())

    u_e_min = np.min(u_pred_n - u_exact_n)
    u_e_max = np.max(u_pred_n - u_exact_n)

    e_max = max(e_max, u_e_max)
    e_min = min(e_min, u_e_min)

levels_u = np.linspace(u_min, u_max, N_LEVELS)

e_levels = np.linspace(e_min, e_max, N_LEVELS)

e_norm = TwoSlopeNorm(
    vmin=e_min,
    vcenter=0.0,
    vmax=e_max
)

#%%

# ------------------------------------------------------------
# Contour plots: exact, prediction, error
# ------------------------------------------------------------

def plot_npz_file(file):
    data = np.load(file)

    Xn = data["Xn"]
    Yn = data["Yn"]

    u_pred_n = data["u_pred_n"]

    u_exact_n = data["u_exact_n"]

    fields = [
        ("u", u_exact_n, u_pred_n, levels_u)
    ]

    for name, exact, pred, levels in fields:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4),
            constrained_layout=True
        )

        fig.suptitle(f"{plot_name[file]}")

        cf1 = axes[0].contourf(Xn, Yn, exact, levels=levels)
        axes[0].set_title(f"Eksakt løsning {name}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(cf1, ax=axes[0])

        cf2 = axes[1].contourf(Xn, Yn, pred, levels=levels)
        axes[1].set_title(f"Gennemsnitlig prædiktion {name}")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(cf2, ax=axes[1])

        error = pred - exact

        cf3 = axes[2].contourf(
            Xn,
            Yn,
            error,
            levels=e_levels,
            norm=e_norm,
            cmap="coolwarm"
        )
        axes[2].set_title(f"Fejl {name}: prædiktion - eksakt")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(cf3, ax=axes[2])

        plt.show()

        L_max = np.max(np.abs(error))
        L_2 = np.linalg.norm(error)

        print(f"{file} | {name} L_max: {L_max:.2e}")
        print(f"{file} | {name} L_2:    {L_2:.2e}")


plot_npz_file(baseline_file)

for file in files:
    plot_npz_file(file)

#%%

# ------------------------------------------------------------
# Final error per seed: L_inf and L2, plotted with baseline
# ------------------------------------------------------------

seeds = np.arange(len(baseline_data["L_max"]))

components = [
    ("u", "L_max", "L_2")
]

for comp_name, key_lmax, key_l2 in components:
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    # Baseline
    color = model_colors[baseline_file]

    ax[0].plot(
        seeds,
        baseline_data[key_lmax],
        label=f"{plot_name[baseline_file]}",
        linewidth=2.5,
        color=color
    )
    ax[0].axhline(
        np.mean(baseline_data[key_lmax]),
        linestyle="--",
        label=f"Gennemsnit {plot_name[baseline_file]}",
        color=color
    )

    ax[1].plot(
        seeds,
        baseline_data[key_l2],
        label=f"{plot_name[baseline_file]}",
        linewidth=2.5,
        color=color
    )
    ax[1].axhline(
        np.mean(baseline_data[key_l2]),
        linestyle="--",
        label=f"Gennemsnit {plot_name[baseline_file]}",
        color=color
    )

    # Other files
    for file, data in zip(files, datasets):
        color = model_colors[file]

        ax[0].plot(
            seeds,
            data[key_lmax],
            label=f"{plot_name[file]}",
            color=color
        )
        ax[0].axhline(
            np.mean(data[key_lmax]),
            linestyle="--",
            label=f"Gennemsnit {plot_name[file]}",
            color=color
        )

        ax[1].plot(
            seeds,
            data[key_l2],
            label=f"{plot_name[file]}",
            color=color
        )
        ax[1].axhline(
            np.mean(data[key_l2]),
            linestyle="--",
            label=f"Gennemsnit {file}",
            color=color
        )

    ax[0].set_xlabel("Seed")
    ax[0].set_ylabel(r"$L_\infty$ fejl")
    ax[0].set_title(fr"{comp_name}: $L_\infty$ vs 'seed'")
    ax[0].grid(True)

    ax[1].set_xlabel("Seed")
    ax[1].set_ylabel(r"$L_2$ fejl")
    ax[1].set_title(fr"{comp_name}: $L_2$ vs 'seed'")
    ax[1].grid(True)

    # Shared legend under both subplots
    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3
    )

    fig.tight_layout(rect=[0, 0.18, 1, 1])

    plt.show()

# %%

# ------------------------------------------------------------
# Validation errors vs epoch
# ------------------------------------------------------------

val_components = [
    ("u", "val_l2", "val_lmax")
]

for comp_name, key_l2, key_lmax in val_components:
    epochs = np.arange(len(baseline_data[key_l2]))

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        constrained_layout=True
    )

    # Baseline
    ax[0].semilogy(
        epochs,
        baseline_data[key_lmax],
        label=f"{plot_name[baseline_file]}",
        linewidth=2.5
    )
    ax[1].semilogy(
        epochs,
        baseline_data[key_l2],
        label=f"{plot_name[baseline_file]}",
        linewidth=2.5
    )

    # Other files
    for file, data in zip(files, datasets):
        ax[0].semilogy(epochs, data[key_lmax], label=f"{plot_name[file]}")
        ax[1].semilogy(epochs, data[key_l2], label=f"{plot_name[file]}")

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(r"$L_\infty$-valideringsfejl")
    ax[0].set_title(fr"{comp_name}: validering $L_\infty$")
    ax[0].grid(True, which="both")
    ax[0].legend()

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"$L_2$-valideringsfejl")
    ax[1].set_title(fr"{comp_name}: validering $L_2$")
    ax[1].grid(True, which="both")
    ax[1].legend()

    plt.show()

#%%

# ------------------------------------------------------------
# Bar plots of mean final norms for each variable
# ------------------------------------------------------------

model_files = [baseline_file] + files
model_data = [baseline_data] + datasets
model_labels = model_files

components = ["u"]
x = np.arange(len(components))
width = 0.25

# Mean L_inf values
fig, ax = plt.subplots(figsize=(9, 4))

for i, (label, data) in enumerate(zip(model_labels, model_data)):
    mean_lmax = [
        np.mean(data["L_max"])
        for comp in components
    ]

    ax.bar(
        x + (i - 1) * width,
        mean_lmax,
        width,
        label=plot_name[label],
        color=model_colors[model_files[i]]
    )

ax.set_xlabel("Variable")
ax.set_ylabel(r"Gennemsnitlig $L_\infty$-fejl")
ax.set_title(r"Gennemsnitlige $L_\infty$-fejl")
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.grid(True, axis="y")
ax.legend()

plt.tight_layout()
plt.show()


# Mean L2 values
fig, ax = plt.subplots(figsize=(9, 4))

for i, (label, data) in enumerate(zip(model_labels, model_data)):
    mean_l2 = [
        np.mean(data["L_2"])
        for comp in components
    ]

    ax.bar(
        x + (i - 1) * width,
        mean_l2,
        width,
        label=plot_name[label],
        color=model_colors[model_files[i]]
    )

ax.set_xlabel("Variabel")
ax.set_ylabel(r"Gennemsnitlig $L_2$-fejl")
ax.set_title(r"Gennemsnitlige $L_2$-fejl")
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.grid(True, axis="y")
ax.legend()

plt.tight_layout()
plt.show()

#%%

# ------------------------------------------------------------
# Print summary table
# ------------------------------------------------------------

print("\nSummary of mean final errors")
print("-" * 70)
print(f"{'Model':<20} {'Komponent':<10} {'L_inf':<15} {'L2':<15}")
print("-" * 70)

for file, data in [(baseline_file, baseline_data)] + list(zip(files, datasets)):
    for comp_name in ["u"]:
        mean_lmax = np.mean(data["L_max"])
        mean_l2 = np.mean(data["L_2"])

        print(
            f"{plot_name[file]:<20} "
            f"{comp_name:<10} "
            f"{mean_lmax:<15.4e} "
            f"{mean_l2:<15.4e}"
        )

print("-" * 70)

# %%

# ------------------------------------------------------------
# Contour plots of mean gate weights for MoE model
# ------------------------------------------------------------

moe_files = [
    "moe2dpos_expanded.npz"
]

for file in moe_files:
    data = np.load(file)

    Xn = data["Xn"]
    Yn = data["Yn"]
    gate_weights = data["gate_weights_n"]

    # gate_weights should have shape: (nx, ny, num_experts)
    num_experts = gate_weights.shape[2]

    vmin = np.min(gate_weights)
    vmax = np.max(gate_weights)

    fig, axes = plt.subplots(
        1,
        num_experts,
        figsize=(4 * num_experts, 4),
        constrained_layout=True
    )

    # If num_experts = 1, axes is not automatically a list
    axes = np.atleast_1d(axes)

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
            f"{plot_name[file]}\nGennemsnits vægte ekspert {k + 1}"
        )
        axes[k].set_xlabel(r"$x$")
        axes[k].set_ylabel(r"$y$")

    fig.colorbar(cf, ax=axes)
    plt.show()
# %%
