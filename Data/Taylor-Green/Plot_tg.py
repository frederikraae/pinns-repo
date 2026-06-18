#%%

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
baseline_file = "tg_pinn_50.npz"

# Files to compare against baseline
files = [
    "tg_moe_50.npz",
    "tg_pinn_soft_50.npz"
]

model_colors = {
    baseline_file: "tab:blue",
    "tg_moe_50.npz": "tab:orange",
    "tg_pinn_soft_50.npz": "tab:green",
}

plot_name = {
    baseline_file: "Naiv PINN",
    "tg_moe_50.npz": "MoE-PINN",
    "tg_pinn_soft_50.npz": "PINN med SoftAdapt",
}

N_LEVELS = 15

# Load datasets
baseline_data = np.load(baseline_file)
datasets = [np.load(file) for file in files]

# p mean correction
P_MEAN = False

#%%
# ------------------------------------------------------------
# Find common color limits for contour plots
# ------------------------------------------------------------

all_datasets = [baseline_data] + datasets

u_min, u_max = np.inf, -np.inf
v_min, v_max = np.inf, -np.inf
p_min, p_max = np.inf, -np.inf
e_min, e_max = np.zeros(len(all_datasets)), np.zeros(len(all_datasets))

for data in all_datasets:
    u_pred_n = data["u_pred_n"]
    v_pred_n = data["v_pred_n"]
    p_pred_n = data["p_pred_n"]

    u_exact_n = data["u_exact_n"]
    v_exact_n = data["v_exact_n"]
    p_exact_n = data["p_exact_n"]

    # p mean correction
    if P_MEAN:
        p_pred_n = p_pred_n - p_pred_n.mean() + p_exact_n.mean()

    u_min = min(u_min, u_pred_n.min(), u_exact_n.min())
    u_max = max(u_max, u_pred_n.max(), u_exact_n.max())

    v_min = min(v_min, v_pred_n.min(), v_exact_n.min())
    v_max = max(v_max, v_pred_n.max(), v_exact_n.max())

    p_min = min(p_min, p_pred_n.min(), p_exact_n.min())
    p_max = max(p_max, p_pred_n.max(), p_exact_n.max())

    u_e_min = np.min(u_pred_n - u_exact_n)
    v_e_min = np.min(v_pred_n - v_exact_n)
    p_e_min = np.min(p_pred_n - p_exact_n)

    u_e_max = np.max(u_pred_n - u_exact_n)
    v_e_max = np.max(v_pred_n - v_exact_n)
    p_e_max = np.max(p_pred_n - p_exact_n)

    e_min[0] = min(e_min[0], u_e_min)
    e_min[1] = min(e_min[1], v_e_min)
    e_min[2] = min(e_min[2], p_e_min)

    e_max[0] = max(e_max[0], u_e_max)
    e_max[1] = max(e_max[1], v_e_max)
    e_max[2] = max(e_max[2], p_e_max)

levels_u = np.linspace(u_min, u_max, N_LEVELS)
levels_v = np.linspace(v_min, v_max, N_LEVELS)
levels_p = np.linspace(p_min, p_max, N_LEVELS)

u_e_min = e_min[0]
v_e_min = e_min[1]
p_e_min = e_min[2]

u_e_max = e_max[0]
v_e_max = e_max[1]
p_e_max = e_max[2]

u_e_levels = np.linspace(u_e_min, u_e_max, N_LEVELS)
v_e_levels = np.linspace(v_e_min, v_e_max, N_LEVELS)
p_e_levels = np.linspace(p_e_min, p_e_max, N_LEVELS)

u_e_norm = TwoSlopeNorm(
    vmin=u_e_min,
    vcenter=0.0,
    vmax=u_e_max
)

v_e_norm = TwoSlopeNorm(
    vmin=v_e_min,
    vcenter=0.0,
    vmax=v_e_max
)

p_e_norm = TwoSlopeNorm(
    vmin=p_e_min,
    vcenter=0.0,
    vmax=p_e_max
)

level = {
    "u": u_e_levels,
    "v": v_e_levels,
    "p": p_e_levels
}

norm = {
    "u": u_e_norm,
    "v": v_e_norm,
    "p": p_e_norm
}

#%%

# ------------------------------------------------------------
# Contour plots: exact, prediction, error
# ------------------------------------------------------------

def plot_npz_file(file):
    data = np.load(file)

    Xn = data["Xn"]
    Yn = data["Yn"]

    u_pred_n = data["u_pred_n"]
    v_pred_n = data["v_pred_n"]
    p_pred_n = data["p_pred_n"]

    u_exact_n = data["u_exact_n"]
    v_exact_n = data["v_exact_n"]
    p_exact_n = data["p_exact_n"]

    # p mean correction
    if P_MEAN:
        p_pred_n = p_pred_n - p_pred_n.mean() + p_exact_n.mean()

    fields = [
        ("u", u_exact_n, u_pred_n, levels_u),
        ("v", v_exact_n, v_pred_n, levels_v),
        ("p", p_exact_n, p_pred_n, levels_p),
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
        axes[0].set_title(f"Eksakt løsning ${name}$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(cf1, ax=axes[0])

        cf2 = axes[1].contourf(Xn, Yn, pred, levels=levels)
        axes[1].set_title(f"Gennemsnitlig prædiktion $\hat{{{name}}}$")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(cf2, ax=axes[1])

        error = pred - exact

        cf3 = axes[2].contourf(
            Xn,
            Yn,
            error,
            levels=level[name],
            norm=norm[name],
            cmap="coolwarm"
        )
        axes[2].set_title(rf"Fejl: $\hat{{{name}}} - {{{name}}}$")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(cf3, ax=axes[2])

        plt.show()

        L_max = np.max(np.abs(error))
        L_2 = np.linalg.norm(error)

        print(f"{file} | {name} L_max: {L_max:.2e}")
        print(f"{file} | {name} L_2:    {L_2:.2e}")
        print(f"{file} | {name} RMSE:    {np.sqrt(np.mean(error**2)):.2e}")
        print(f"{file} | {name} error variance:    {error.var():.2e}")
        


plot_npz_file(baseline_file)

for file in files:
    plot_npz_file(file)

#%%

# ------------------------------------------------------------
# Final error per seed: L_inf and L2, plotted with baseline
# ------------------------------------------------------------

seeds = np.arange(len(baseline_data["u_L_max"]))

components = [
    ("u", "u_L_max", "u_L_2"),
    ("v", "v_L_max", "v_L_2"),
    ("p", "p_L_max", "p_L_2"),
]

for comp_name, key_lmax, key_l2 in components:
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5)
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
    ax[0].set_title(fr"$\hat{{{comp_name}}}$: $L_\infty$ vs 'seed'")
    ax[0].grid(True)

    ax[1].set_xlabel("Seed")
    ax[1].set_ylabel(r"$L_2$ fejl")
    ax[1].set_title(fr"$\hat{{{comp_name}}}$: $L_2$ vs 'seed'")
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

#%%

# ------------------------------------------------------------
# Validation errors vs epoch
# ------------------------------------------------------------

val_components = [
    ("u", "val_u_l2", "val_u_lmax"),
    ("v", "val_v_l2", "val_v_lmax"),
    ("p", "val_p_l2", "val_p_lmax"),
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
        ax[0].semilogy(epochs, data[key_lmax], label=plot_name[file])
        ax[1].semilogy(epochs, data[key_l2], label=plot_name[file])

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(r"$L_\infty$-valideringsfejl")
    ax[0].set_title(fr"$\hat{{{comp_name}}}$: validering $L_\infty$")
    ax[0].grid(True, which="both")
    ax[0].legend()

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"$L_2$-valideringsfejl")
    ax[1].set_title(fr"$\hat{{{comp_name}}}$: validering $L_2$")
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

components = ["u", "v", "p"]
x = np.arange(len(components))
width = 0.25

# Mean L_inf values
fig, ax = plt.subplots(figsize=(9, 4))

for i, (label, data) in enumerate(zip(model_labels, model_data)):
    mean_lmax = [
        np.mean(data[f"{comp}_L_max"])
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
        np.mean(data[f"{comp}_L_2"])
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

print("\nSummary of final errors")
print("-" * 70)
print(f"{'Model':<20} {'Komponent':<10} {'L_inf':<20} {'L2':<20}")
print(f"{' ':<31} {'Mean':<10}{'Std':<10} {'Mean':<10}{'Std':<10}" )
print("-" * 70)

for file, data in [(baseline_file, baseline_data)] + list(zip(files, datasets)):
    for comp_name in ["u", "v", "p"]:
        mean_lmax = np.mean(data[f"{comp_name}_L_max"])
        std_lmax = np.std(data[f"{comp_name}_L_max"])
        mean_l2 = np.mean(data[f"{comp_name}_L_2"])
        std_l2 = np.std(data[f"{comp_name}_L_2"])

        print(
            f"{plot_name[file]:<20} "
            f"{comp_name:<10} "
            f"{mean_lmax:<10.2e}"
            f"{std_lmax:<10.2e} "
            f"{mean_l2:<10.2e}"
            f"{std_l2:<10.2e}"
        )

print("-" * 70)

# %%

# ------------------------------------------------------------
# Contour plots of mean gate weights for MoE model
# ------------------------------------------------------------

moe_files = [
    "tg_moe_50.npz",
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












