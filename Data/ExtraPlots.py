import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("moe2dpos_expanded.npz")

L_max = data1["L_max"]
L_2 = data1["L_2"]

seeds = np.arange(len(L_max))

L_max_mean = np.mean(L_max)
L_2_mean = np.mean(L_2)

plt.figure(figsize=(9, 5))

plt.plot(seeds, L_max, marker="o", label=r"$L_\infty$")
plt.plot(seeds, L_2, marker="s", label=r"$L_2$")

plt.axhline(L_max_mean, linestyle="--", label=fr"Mean $L_\infty$ = {L_max_mean:.2e}")
plt.axhline(L_2_mean, linestyle="--", label=fr"Mean $L_2$ = {L_2_mean:.2e}")

plt.xlabel("Seed")
plt.ylabel("Error norm")
plt.title(r"$L_2$ and $L_\infty$ error norms vs seed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()