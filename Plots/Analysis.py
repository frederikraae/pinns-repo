import numpy as np
import matplotlib.pyplot as plt

base = np.load("base.npz")
softa = np.load("softa.npz")
rar = np.load("rar.npz")
data = np.load("data.npz")

# Plot solution
plt.figure()
plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(t_test.numpy(), x_pred.numpy(), "--", label="PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Van der Pol oscillator")
plt.legend()
plt.grid()
plt.show()

#%%
# Training loss
loss_history = np.array(loss_history)
window = 50
loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode="valid")
plt.figure()
plt.plot(loss_smooth)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid()
plt.show()
