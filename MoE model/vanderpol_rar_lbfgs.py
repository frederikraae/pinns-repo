#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from new_model import MoEPINN

torch.manual_seed(0)

#%%
# Parameters
t0 = 0.0
T = 20.0
mu = 5.0

x0 = 2.0
v0 = 0.0

x0_true = torch.tensor([[x0]], dtype=torch.float32)
v0_true = torch.tensor([[v0]], dtype=torch.float32)

#%%
# MoE model: input is only t
moe_model = MoEPINN(
    in_dim=1, 
    out_dim=1, 
    num_experts=3, 
    expert_hidden_dim=32, 
    expert_hidden_layers=2, 
    gate_hidden_dim=16, 
    gate_hidden_layers=2, 
    temperature=0.7
    )
# optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3)

#%%
# Numerical reference solution
def f(t, y):
    x, v = y
    a = mu * (1 - x**2) * v - x
    return [v, a]

t_span = (t0, T)
t_eval = np.linspace(t0, T, 10000)
sol = solve_ivp(f, t_span, [x0, v0], t_eval=t_eval)

#%%
# Residual-based adaptive refinement implementation

def sobol_sample(engine, n, t0, T):
    t = engine.draw(n, dtype=torch.float32)
    return t0 + (T - t0) * t

def vdp_residual(model, t, mu):
    out = model(t)
    x = out[0] if isinstance(out, tuple) else out

    x_t = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    x_tt = torch.autograd.grad(
        x_t, t,
        grad_outputs=torch.ones_like(x_t),
        create_graph=True
    )[0]

    return x_tt - mu * (1 - x**2) * x_t + x

def vdp_ic_loss(model, t0, x0, v0):
    t_ic = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True)

    out = model(t_ic)
    x = out[0] if isinstance(out, tuple) else out

    x_t = torch.autograd.grad(
        x, t_ic,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    loss_x0 = (x - x0)**2
    loss_v0 = (x_t - v0)**2

    return torch.mean(loss_x0 + loss_v0)

def compute_total_loss(model, T_res, t0, x0, v0, mu, lambda_ic=1.0):
    t = T_res.clone().detach().requires_grad_(True)
    R = vdp_residual(model, t, mu)
    loss_pde = torch.mean(R**2)
    loss_ic = vdp_ic_loss(model, t0, x0, v0)
    return loss_pde + lambda_ic * loss_ic

def train_on_res_points(
        model, 
        T_res, 
        n_epochs_adam, 
        t0, 
        x0, 
        v0, 
        mu, 
        lambda_ic=1.0,
        adam_lr=1e-3,
        use_lbfgs=True,
        lbfgs_max_iter=200,
        lbfgs_history_size=50
        ):
    loss_history = []

    # Phase 1: Adam optimization
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)

    for _ in range(n_epochs_inner):
        optimizer_adam.zero_grad()
        loss = compute_total_loss(model, T_res, t0, x0, v0, mu, lambda_ic)
        loss.backward()
        optimizer_adam.step()
        loss_history.append(loss.item())

    # Phase 2: L-BFGS 
    if use_lbfgs:
        optimizer_lfbgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max_iter,
            max_eval=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer_lfbgs.zero_grad()
            loss = compute_total_loss(model, T_res, t0, x0, v0, mu, lambda_ic)
            loss.backward()
            return loss

        optimizer_lfbgs.step(closure)

        final_loss = compute_total_loss(model, T_res, t0, x0, v0, mu, lambda_ic)
        loss_history.append(final_loss.item())
    
    return loss_history

def rar_step(model, T_res, cand_engine, n_cand, m_add, mu, t0, T):
    S0 = sobol_sample(cand_engine, n_cand, t0, T)
    S0.requires_grad_(True)

    R0 = vdp_residual(model, S0, mu)
    score = R0.abs().detach().squeeze()

    idx = torch.topk(score, k=m_add).indices
    S = S0[idx].detach()

    T_res = torch.cat([T_res, S], dim=0)
    T_res, _ = torch.sort(T_res, dim=0)
    return T_res

# Training parameters 
N_init = 256
n_epochs_inner = 1000
n_rar_iter = 20
N_cand = 5000
m_add = 20
lbfgs_max_iter = 100
lbfgs_history_size = 50

init_engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
cand_engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

T_res = sobol_sample(init_engine, N_init, t0, T).detach()

loss_history = []

block_loss = train_on_res_points(
    model=moe_model, 
    T_res=T_res, 
    n_epochs_adam=n_epochs_inner, 
    t0=t0, 
    x0=x0_true, 
    v0=v0_true, 
    mu=mu,
    lambda_ic=3.0,
    adam_lr=1e-3,
    use_lbfgs=True,
    lbfgs_max_iter=lbfgs_max_iter,
    lbfgs_history_size=lbfgs_history_size
    )
loss_history.extend(block_loss)

for rar_iter in range(n_rar_iter):
    T_res = rar_step(moe_model, T_res, cand_engine, N_cand, m_add, mu, t0, T)

    block_loss = train_on_res_points(
        model=moe_model, 
        T_res=T_res, 
        n_epochs_adam=n_epochs_inner, 
        t0=t0, 
        x0=x0_true, 
        v0=v0_true, 
        mu=mu,
        lambda_ic=1.0,
        adam_lr=1e-3,
        use_lbfgs=True,
        lbfgs_max_iter=200,
        lbfgs_history_size=50
    )
    loss_history.extend(block_loss)

    print(f"Outer itereaton: {rar_iter}. Loss={loss_history[-1]}")

#%%
# Evaluation
N_test = 2000
moe_model.eval()

# Test grid
t_test = torch.linspace(t0, T, N_test, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    x_pred, gate_pred, _, _ = moe_model(t_test)

x_pred = x_pred.detach()
gate_pred = gate_pred.detach()
t_test = t_test.detach()

# Residual grid
t_res = torch.linspace(t0, T, N_test, dtype=torch.float32).view(-1, 1)
t_res.requires_grad_(True)

R = vdp_residual(moe_model, t_res, mu)
R_abs = R.abs().detach().numpy()
t_np = t_res.detach().numpy()


#%%
# Plot solution
plt.figure()
plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(t_test.numpy(), x_pred.numpy(), "--", label="MoE-PINN")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Van der Pol oscillator")
plt.legend()
plt.grid()
plt.show()

#%%
# Plot gate responses
plt.figure()
for k in range(moe_model.num_experts):
    plt.plot(t_test.numpy(), gate_pred[:, k].numpy(), label=f"Expert {k+1}")
plt.xlabel("t")
plt.ylabel("Gate weight")
plt.title("Gate responses over time")
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

#%%
# Residual vs t plot
plt.figure(figsize=(8,4))
plt.plot(t_np, R_abs)
plt.xlabel("t")
plt.ylabel("|R(t)|")
plt.title("Residual vs t")
plt.yscale("log")   # very useful
plt.grid(True)
plt.show()
#%%