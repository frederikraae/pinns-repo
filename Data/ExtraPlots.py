import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("moe2dpos_expanded.npz")

L_max = data1["L_max"]
L_2 = data1["L_2"]

seeds = np.arange(len(L_max))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(seeds, L_max, marker="o", label=r"$L_\infty$")
ax[0].axhline(np.mean(L_max), linestyle="--", label=fr"Mean = {np.mean(L_max):.2e}")
ax[0].set_xlabel("Seed")
ax[0].set_ylabel(r"$L_\infty$ error")
ax[0].set_title(r"$L_\infty$ vs seed")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(seeds, L_2, marker="o", label=r"$L_2$")
ax[1].axhline(np.mean(L_2), linestyle="--", label=fr"Mean = {np.mean(L_2):.2e}")
ax[1].set_xlabel("Seed")
ax[1].set_ylabel(r"$L_2$ error")
ax[1].set_title(r"$L_2$ vs seed")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()