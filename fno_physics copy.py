import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuralop.models import FNO
from network import MLP

torch.manual_seed(0)

# ============================================================
# 1. Make dataset for Poisson: -u''(x) = f(x), x in [0,L]
# ============================================================
n_samples = 200
n_grid = 128
n_modes = 8
L = 10.0

x_grid = torch.linspace(0, L, n_grid)                  # (grid,)
dx = x_grid[1] - x_grid[0]

f_data = torch.zeros(n_samples, n_grid)
u_data = torch.zeros(n_samples, n_grid)

# Build synthetic dataset:
# f(x) = sum_k a_k sin(k pi x / L)
# u(x) = sum_k a_k / (k pi / L)^2 sin(k pi x / L)
# so that -u''(x) = f(x)
for k in range(1, n_modes + 1):
    a_k = torch.randn(n_samples, 1)                    # (samples,1)
    basis = torch.sin(k * np.pi * x_grid / L).view(1, n_grid)

    f_data += a_k * basis                           
    u_data += a_k / ((k * np.pi / L) ** 2) * basis    #!!! SKAL LØSES NUMERISK FOR FOURIER !!!

f_data = f_data.unsqueeze(1)                           # (samples,1,grid)
u_data = u_data.unsqueeze(1)                           # (samples,1,grid)

x_channel = x_grid.view(1, 1, n_grid).repeat(n_samples, 1, 1)
X = torch.cat([f_data, x_channel], dim=1)             # (samples,2,grid)
Y = u_data                                            # (samples,1,grid)

# ============================================================
# 2. Normalize target for FNO training
# ============================================================
y_mean = Y.mean()
y_std = Y.std()
Y_norm = (Y - y_mean) / y_std

# ============================================================
# 3. Train FNO on data
# ============================================================
fno = FNO(
    n_modes=(16,),
    hidden_channels=32,
    in_channels=2,
    out_channels=1
)

opt_fno = torch.optim.Adam(fno.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

n_epochs_fno = 1000

for epoch in range(n_epochs_fno):
    pred = fno(X)
    loss = loss_fn(pred, Y_norm)

    opt_fno.zero_grad()
    loss.backward()
    opt_fno.step()

    if epoch % 20 == 0:
        print(f"[FNO ] epoch {epoch:3d} | loss = {loss.item():.6e}")

# Freeze FNO
for p in fno.parameters():
    p.requires_grad = False
fno.eval()

# ============================================================
# 4. Pick one sample to correct with PINN
# ============================================================
i = 0

X_sample = X[i:i+1]                                   # (1,2,grid)
f_sample_vals = f_data[i, 0]                          # (grid,)
u_true_vals = Y[i, 0]                                 # (grid,)

with torch.no_grad():
    u_fno_norm = fno(X_sample)                        # (1,1,grid)
    u_fno_vals = (u_fno_norm * y_std + y_mean).squeeze()  # (grid,)

# ============================================================
# 5. Helper: 1D linear interpolation
# ============================================================
def interp_from_grid(x_query, x_grid, y_grid):
    """
    x_query: (N,1)
    x_grid : (G,)
    y_grid : (G,)
    returns: (N,1)
    """
    xq = x_query.squeeze(-1)
    xq = torch.clamp(xq, x_grid[0], x_grid[-1])

    idx = torch.searchsorted(x_grid, xq)
    idx = torch.clamp(idx, 1, len(x_grid) - 1)

    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    y0 = y_grid[idx - 1]
    y1 = y_grid[idx]

    t = (xq - x0) / (x1 - x0)
    yq = y0 + t * (y1 - y0)

    return yq.unsqueeze(-1)

# ============================================================
# 6. Approximate FNO second derivative on the grid
# ============================================================
u_fno_xx_grid = torch.zeros_like(u_fno_vals)

# central difference for interior points
u_fno_xx_grid[1:-1] = (
    u_fno_vals[:-2] - 2 * u_fno_vals[1:-1] + u_fno_vals[2:]
) / dx**2

# simple boundary fill
u_fno_xx_grid[0] = u_fno_xx_grid[1]
u_fno_xx_grid[-1] = u_fno_xx_grid[-2]

# ============================================================
# 7. PINN correction network
# ============================================================
pinn = MLP(1, 1, [64, 64, 64, 64])
opt_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-3)

xa = torch.tensor([[0.0]], dtype=torch.float32)
xb = torch.tensor([[L]], dtype=torch.float32)

# In this synthetic sine-series setup, u(0)=u(L)=0
u_a_true = torch.tensor([[0.0]], dtype=torch.float32)
u_b_true = torch.tensor([[0.0]], dtype=torch.float32)

# ============================================================
# 8. Train PINN as residual corrector
# ============================================================
lam_bc = 10.0
lam_reg = 1
n_epochs_pinn = 1000

for epoch in range(n_epochs_pinn):
    opt_pinn.zero_grad()

    # collocation points
    x_phys = torch.linspace(0, L, 200).unsqueeze(1)
    x_phys.requires_grad_(True)

    # frozen FNO coarse solution and its approx 2nd derivative at collocation points
    u_fno_phys = interp_from_grid(x_phys, x_grid, u_fno_vals)
    u_fno_xx_phys = interp_from_grid(x_phys, x_grid, u_fno_xx_grid)

    # forcing at collocation points
    f_phys = interp_from_grid(x_phys, x_grid, f_sample_vals)

    # PINN correction
    u_corr = pinn(x_phys)

    # total solution
    u_pred = u_fno_phys + u_corr

    # differentiate only correction
    u_corr_x = torch.autograd.grad(
        u_corr, x_phys,
        grad_outputs=torch.ones_like(u_corr),
        create_graph=True
    )[0]

    u_corr_xx = torch.autograd.grad(
        u_corr_x, x_phys,
        grad_outputs=torch.ones_like(u_corr_x),
        create_graph=True
    )[0]

    # PDE residual for -u'' = f  <=>  u'' + f = 0
    # Using u = u_fno + u_corr:
    # residual = u_fno_xx + u_corr_xx + f
    residual = u_fno_xx_phys + u_corr_xx + f_phys
    loss_pde = torch.mean(residual**2)

    # boundary loss
    u_fno_a = interp_from_grid(xa, x_grid, u_fno_vals)
    u_fno_b = interp_from_grid(xb, x_grid, u_fno_vals)

    u_pred_a = u_fno_a + pinn(xa)
    u_pred_b = u_fno_b + pinn(xb)

    loss_bc = torch.mean((u_pred_a - u_a_true)**2 + (u_pred_b - u_b_true)**2)

    # keep correction small
    loss_reg = torch.mean(u_corr**2)

    loss = loss_pde + lam_bc * loss_bc + lam_reg * loss_reg

    loss.backward()
    opt_pinn.step()

    if epoch % 200 == 0:
        print(
            f"[PINN] epoch {epoch:4d} | "
            f"loss={loss.item():.6e} | "
            f"pde={loss_pde.item():.6e} | "
            f"bc={loss_bc.item():.6e} | "
            f"reg={loss_reg.item():.6e}"
        )

# ============================================================
# 9. Evaluate
# ============================================================
pinn.eval()

x_test = x_grid.unsqueeze(1)

with torch.no_grad():
    u_corr_test = pinn(x_test).squeeze()
    u_hybrid = u_fno_vals + u_corr_test

# ============================================================
# 10. Plot
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(x_grid.numpy(), f_sample_vals.numpy(), "--", label="f(x)")
plt.plot(x_grid.numpy(), u_true_vals.numpy(), label="true u(x)")
plt.plot(x_grid.numpy(), u_fno_vals.numpy(), label="FNO")
plt.plot(x_grid.numpy(), u_hybrid.numpy(), label="FNO + PINN correction")
plt.xlabel("x")
plt.ylabel("value")
plt.title("Hybrid solver: FNO + PINN")
plt.legend()
plt.tight_layout()
plt.show()

# pointwise squared error for the trained sample
se_fno = (u_fno_vals - u_true_vals) ** 2
se_hybrid = (u_hybrid - u_true_vals) ** 2

plt.figure(figsize=(8, 5))
plt.plot(x_grid.numpy(), se_fno.numpy(), label="FNO squared error")
plt.plot(x_grid.numpy(), se_hybrid.numpy(), label="Hybrid squared error")
plt.xlabel("x")
plt.ylabel("Squared error")
plt.title("Pointwise squared error on trained sample")
plt.legend()
plt.tight_layout()
plt.show()