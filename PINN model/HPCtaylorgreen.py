import os
import sys
import numpy as np
import torch
from torch import sin, cos, exp, pi
import multiprocessing as mp
from time import perf_counter

from network import PINN
from softadapt import LossWeightedSoftAdapt

# Parameters
V0 = 1.0
L = 1.0
rho = 1.0
nu = 0.01

T = 10

# Analytical solution
u_analytical = lambda x, y, t: V0 * cos(x / L) * sin(y / L) * exp(-2 * (nu / L**2) * t)
v_analytical = lambda x, y, t: -V0 * sin(x / L) * cos(y / L) * exp(-2 * (nu / L**2) * t)
p_analytical = lambda x, y, t: (-rho / 4) * V0**2 * (cos(2*x / L) + cos(2*y / L)) * exp(-4 * (nu / L**2) * t)


def run_seed(args):
    seed, w_softa = args

    print(f'Seed {seed} is running')
     # Avoid each process using too many CPU threads
    torch.set_num_threads(1)

    torch.manual_seed(seed)

    # Initialize network and define optimizer
    net = PINN(
        in_dim=3,
        out_dim=3,
        hidden_dim=128,
        hidden_layers=8
    )

    # Initial points
    N_ic = 2000
    x_ic = - (L * pi) + 2 * L * pi * torch.rand(N_ic, 1)
    y_ic = - (L * pi) + 2 * L * pi * torch.rand(N_ic, 1)
    t_ic = torch.zeros_like(x_ic)
    X_ic = torch.cat([x_ic,y_ic,t_ic], dim=1)

    # Fixed validation grid

    n_val = 20

    x_val = torch.linspace(-pi, pi, n_val)
    y_val = torch.linspace(-pi, pi, n_val)

    X_val, Y_val = torch.meshgrid(x_val, y_val, indexing='ij')

    t_val = torch.full_like(X_val.reshape(-1,1), T/2)

    XYT_val = torch.cat(
        [X_val.reshape(-1,1), Y_val.reshape(-1,1), t_val],
        dim=1
    )

    u_true_val = u_analytical(XYT_val[:,0:1], XYT_val[:,1:2], XYT_val[:,2:3])
    v_true_val = v_analytical(XYT_val[:,0:1], XYT_val[:,1:2], XYT_val[:,2:3])
    p_true_val = p_analytical(XYT_val[:,0:1], XYT_val[:,1:2], XYT_val[:,2:3])

    val_exact = (u_true_val, v_true_val, p_true_val)

    norms = [[[], []],
             [[], []],
             [[], []]]

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    if w_softa:
        softadapt_object = LossWeightedSoftAdapt(beta=0.1, accuracy_order=5)
        window = 25
        loss_hist_1 = []
        loss_hist_2 = []
        loss_hist_3 = []

    # Loss term weights
    lam_pde = 3.0
    lam_bc = 5.0
    lam_ic = 5.0

    # Interior points (collacation)
    N = 50

    # Training loop
    n_epoch = 10_000

    for epoch in range(n_epoch):
        optimizer.zero_grad()

        # Collocation points
        x = - (L * pi) + 2 * L * pi * torch.rand(N, 1)
        y = - (L * pi) + 2 * L * pi * torch.rand(N, 1)
        t = T * torch.rand(N, 1)
        X = torch.cat([x,y,t], dim=1)
        
        X.requires_grad_(True)

        # Boundary points
        pi_like = pi * torch.ones_like(x)
        X_bc_left = torch.cat([-pi_like,y,t], dim=1)
        X_bc_right = torch.cat([pi_like,y,t], dim=1)
        X_bc_bottom = torch.cat([x,-pi_like,t], dim=1)
        X_bc_top = torch.cat([x,pi_like,t], dim=1)

        X_bc_left.requires_grad_(True)
        X_bc_right.requires_grad_(True)
        X_bc_bottom.requires_grad_(True)
        X_bc_top.requires_grad_(True)

        # Forward pass
        G = net(X)

        u = G[:, 0:1]
        v = G[:, 1:2]
        p = G[:, 2:3]

        grad_u = torch.autograd.grad(
            u, X,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        grad_v = torch.autograd.grad(
            v, X,
            grad_outputs=torch.ones_like(v),
            create_graph=True
        )[0]

        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        v_t = grad_v[:, 2:3]

        grad_p = torch.autograd.grad(
            p, X,
            grad_outputs=torch.ones_like(p),
            create_graph=True
        )[0]

        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]


        u_xx = torch.autograd.grad(
            u_x, X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:,0:1]

        u_yy = torch.autograd.grad(
            u_y, X,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:,1:2]

        v_xx = torch.autograd.grad(
            v_x, X,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True
        )[0][:,0:1]

        v_yy = torch.autograd.grad(
            v_y, X,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True
        )[0][:,1:2]


        # PDE loss
        loss_pde_eq1 = torch.mean((u_x + v_y)**2)
        loss_pde_eq2 = torch.mean((u_t + u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy))**2) 
        loss_pde_eq3 = torch.mean((v_t + u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy))**2)
        
        loss_pde = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3

        # BC loss
        G_bc_left = net(X_bc_left)
        G_bc_right = net(X_bc_right)
        G_bc_bottom = net(X_bc_bottom)
        G_bc_top = net(X_bc_top)

        u_bc_left = G_bc_left[:, 0:1]
        v_bc_left = G_bc_left[:, 1:2]
        p_bc_left = G_bc_left[:, 2:3]

        u_bc_right = G_bc_right[:, 0:1]
        v_bc_right = G_bc_right[:, 1:2]
        p_bc_right = G_bc_right[:, 2:3]

        u_bc_bottom = G_bc_bottom[:, 0:1]
        v_bc_bottom = G_bc_bottom[:, 1:2]
        p_bc_bottom = G_bc_bottom[:, 2:3]

        u_bc_top = G_bc_top[:, 0:1]
        v_bc_top = G_bc_top[:, 1:2]
        p_bc_top = G_bc_top[:, 2:3]

        loss_bc_x = torch.mean((u_bc_left - u_bc_right)**2) + torch.mean((v_bc_left - v_bc_right)**2) + torch.mean((p_bc_left - p_bc_right)**2)
        loss_bc_y = torch.mean((u_bc_bottom - u_bc_top)**2) + torch.mean((v_bc_bottom - v_bc_top)**2) + torch.mean((p_bc_bottom - p_bc_top)**2)

        grad_u_bc_left = torch.autograd.grad(
            u_bc_left, X_bc_left,
            grad_outputs=torch.ones_like(u_bc_left),
            create_graph=True
        )[0]

        u_x_bc_left = grad_u_bc_left[:, 0:1]

        grad_u_bc_right = torch.autograd.grad(
            u_bc_right, X_bc_right,
            grad_outputs=torch.ones_like(u_bc_right),
            create_graph=True
        )[0]

        u_x_bc_right = grad_u_bc_right[:, 0:1]

        grad_u_bc_bottom = torch.autograd.grad(
            u_bc_bottom, X_bc_bottom,
            grad_outputs=torch.ones_like(u_bc_bottom),
            create_graph=True
        )[0]

        u_y_bc_bottom = grad_u_bc_bottom[:, 1:2]

        grad_u_bc_top = torch.autograd.grad(
            u_bc_top, X_bc_top,
            grad_outputs=torch.ones_like(u_bc_top),
            create_graph=True
        )[0]

        u_y_bc_top = grad_u_bc_top[:, 1:2]


        grad_v_bc_left = torch.autograd.grad(
            v_bc_left, X_bc_left,
            grad_outputs=torch.ones_like(v_bc_left),
            create_graph=True
        )[0]

        v_x_bc_left = grad_v_bc_left[:, 0:1]

        grad_v_bc_right = torch.autograd.grad(
            v_bc_right, X_bc_right,
            grad_outputs=torch.ones_like(v_bc_right),
            create_graph=True
        )[0]

        v_x_bc_right = grad_v_bc_right[:, 0:1]

        grad_v_bc_bottom = torch.autograd.grad(
            v_bc_bottom, X_bc_bottom,
            grad_outputs=torch.ones_like(v_bc_bottom),
            create_graph=True
        )[0]

        v_y_bc_bottom = grad_v_bc_bottom[:, 1:2]

        grad_v_bc_top = torch.autograd.grad(
            v_bc_top, X_bc_top,
            grad_outputs=torch.ones_like(v_bc_top),
            create_graph=True
        )[0]

        v_y_bc_top = grad_v_bc_top[:, 1:2]

        grad_p_bc_left = torch.autograd.grad(
            p_bc_left, X_bc_left,
            grad_outputs=torch.ones_like(p_bc_left),
            create_graph=True
        )[0]

        p_x_bc_left = grad_p_bc_left[:, 0:1]

        grad_p_bc_right = torch.autograd.grad(
            p_bc_right, X_bc_right,
            grad_outputs=torch.ones_like(p_bc_right),
            create_graph=True
        )[0]

        p_x_bc_right = grad_p_bc_right[:, 0:1]

        grad_p_bc_bottom = torch.autograd.grad(
            p_bc_bottom, X_bc_bottom,
            grad_outputs=torch.ones_like(p_bc_bottom),
            create_graph=True
        )[0]

        p_y_bc_bottom = grad_p_bc_bottom[:, 1:2]

        grad_p_bc_top = torch.autograd.grad(
            p_bc_top, X_bc_top,
            grad_outputs=torch.ones_like(p_bc_top),
            create_graph=True
        )[0]

        p_y_bc_top = grad_p_bc_top[:, 1:2]

        loss_bc_grad_x = torch.mean((u_x_bc_left - u_x_bc_right)**2) + torch.mean((v_x_bc_left - v_x_bc_right)**2) + torch.mean((p_x_bc_left - p_x_bc_right)**2)
        loss_bc_grad_y = torch.mean((u_y_bc_bottom - u_y_bc_top)**2) + torch.mean((v_y_bc_bottom - v_y_bc_top)**2) + torch.mean((p_y_bc_bottom - p_y_bc_top)**2)

        loss_bc = loss_bc_x + loss_bc_y + loss_bc_grad_x + loss_bc_grad_y
    
        # IC loss
        G_ic = net(X_ic)

        u_ic = G_ic[:, 0:1]
        v_ic = G_ic[:, 1:2]
        p_ic = G_ic[:, 2:3]

        loss_ic_eq1 = torch.mean((u_ic - cos(x_ic) * sin(y_ic))**2)
        loss_ic_eq2 = torch.mean((v_ic + sin(x_ic) * cos(y_ic))**2)
        loss_ic_eq3 = torch.mean((p_ic + (1/4) * (cos(2*x_ic) + cos(2*y_ic)))**2)

        loss_ic = loss_ic_eq1 + loss_ic_eq2 + loss_ic_eq3

        if w_softa:

            loss_hist_1.append(loss_pde.item())
            loss_hist_2.append(loss_bc.item())
            loss_hist_3.append(loss_ic.item())

            if epoch >= window and epoch % window == 0:
                weights = softadapt_object.get_component_weights(
                    torch.tensor(loss_hist_1[-window:], dtype=torch.float32),
                    torch.tensor(loss_hist_2[-window:], dtype=torch.float32),
                    torch.tensor(loss_hist_3[-window:], dtype=torch.float32)
                )
                lam_pde, lam_bc, lam_ic = [w.item() for w in weights]

        # Total loss
        loss = lam_pde * loss_pde + lam_bc * loss_bc + lam_ic * loss_ic

        loss.backward()
        optimizer.step()

        # Validation error on fixed validation grid
        with torch.no_grad():
            G_pred = net(XYT_val)
        
            u_pred = G_pred[:, 0:1]
            v_pred = G_pred[:, 1:2]
            p_pred = G_pred[:, 2:3]

            for i, pred in enumerate((u_pred, v_pred, p_pred)):
                err_val = pred - val_exact[i]

                v_l2 = torch.norm(err_val)
                v_lmax = torch.max(torch.abs(err_val))

                norms[i][0].append(v_l2.item())
                norms[i][1].append(v_lmax.item())

    # Evaluation
    n_test = 200

    x_test = torch.linspace(-pi, pi, n_test)
    y_test = torch.linspace(-pi, pi, n_test)

    X_test, Y_test = torch.meshgrid(x_test, y_test, indexing='ij')

    t_test = torch.full_like(X_test.reshape(-1,1), T/2)

    XYT = torch.cat(
        [X_test.reshape(-1,1), Y_test.reshape(-1,1), t_test],
        dim=1
    )

    net.eval()

    with torch.no_grad():
        G_pred = net(XYT)
        
        u_pred = G_pred[:, 0:1]
        v_pred = G_pred[:, 1:2]
        p_pred = G_pred[:, 2:3]

        u_exact = u_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])
        v_exact = v_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])
        p_exact = p_analytical(XYT[:,0:1], XYT[:,1:2], XYT[:,2:3])

    u_pred = u_pred.reshape(n_test, n_test)
    v_pred = v_pred.reshape(n_test, n_test)
    p_pred = p_pred.reshape(n_test, n_test)

    u_exact = u_exact.reshape(n_test, n_test)
    v_exact = v_exact.reshape(n_test, n_test)
    p_exact = p_exact.reshape(n_test, n_test)

    return {
        "seed" : seed,
        "X" : X_test.numpy(),
        "Y" : Y_test.numpy(),
        "u_pred" : u_pred.numpy(),
        "v_pred" : v_pred.numpy(),
        "p_pred" : p_pred.numpy(),
        "u_exact" : u_exact.numpy(),
        "v_exact" : v_exact.numpy(),
        "p_exact" : p_exact.numpy(),
        "val_u_l2" : norms[0][0],
        "val_u_lmax" : norms[0][1],
        "val_v_l2" : norms[1][0],
        "val_v_lmax" : norms[1][1],
        "val_p_l2" : norms[2][0],
        "val_p_lmax" : norms[2][1],
    }

if __name__ == "__main__":
    max_procs = int(sys.argv[1]) if len(sys.argv) > 1 else int(
        os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())
    )

    NUMBER_OF_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    seeds = list(range(NUMBER_OF_SEEDS))
    n_procs = min(max_procs, len(seeds))

    print(f"--- Running {NUMBER_OF_SEEDS} seeds with {n_procs} processes ---")

    w_softa = sys.argv[3].lower() in ["true", "1", "yes", "y"] if len(sys.argv) > 3 else False

    start = perf_counter()

    with mp.Pool(n_procs) as pool:
        results = pool.map(
            run_seed,
            [(seed, w_softa) for seed in seeds],
            chunksize=1
        )
    elapsed = perf_counter() - start

    n = len(results)

    # Mean fields over all seeds
    Xn = results[0]["X"]
    Yn = results[0]["Y"]

    u_pred_n = np.zeros_like(results[0]["u_pred"])
    v_pred_n = np.zeros_like(results[0]["v_pred"])
    p_pred_n = np.zeros_like(results[0]["p_pred"])

    u_exact_n = np.zeros_like(results[0]["u_exact"])
    v_exact_n = np.zeros_like(results[0]["v_exact"])
    p_exact_n = np.zeros_like(results[0]["p_exact"])

    # Mean errors over all seeds
    u_error_n = np.zeros_like(results[0]["u_pred"])
    v_error_n = np.zeros_like(results[0]["v_pred"])
    p_error_n = np.zeros_like(results[0]["p_pred"])

    # Per-seed final errors
    u_L_max = np.zeros(n)
    v_L_max = np.zeros(n)
    p_L_max = np.zeros(n)

    u_L_2 = np.zeros(n)
    v_L_2 = np.zeros(n)
    p_L_2 = np.zeros(n)

    # Validation histories averaged over seeds
    val_u_l2 = np.zeros_like(np.array(results[0]["val_u_l2"]))
    val_u_lmax = np.zeros_like(np.array(results[0]["val_u_lmax"]))

    val_v_l2 = np.zeros_like(np.array(results[0]["val_v_l2"]))
    val_v_lmax = np.zeros_like(np.array(results[0]["val_v_lmax"]))

    val_p_l2 = np.zeros_like(np.array(results[0]["val_p_l2"]))
    val_p_lmax = np.zeros_like(np.array(results[0]["val_p_lmax"]))

    for i, result in enumerate(results):

        # Average predictions
        u_pred_n += result["u_pred"] / n
        v_pred_n += result["v_pred"] / n
        p_pred_n += result["p_pred"] / n

        # Average exact solutions
        u_exact_n += result["u_exact"] / n
        v_exact_n += result["v_exact"] / n
        p_exact_n += result["p_exact"] / n

        # Errors for this seed
        u_error = result["u_pred"] - result["u_exact"]
        v_error = result["v_pred"] - result["v_exact"]
        p_error = result["p_pred"] - result["p_exact"]

        # Average errors
        u_error_n += u_error / n
        v_error_n += v_error / n
        p_error_n += p_error / n

        # Per-seed final norms
        u_L_max[i] = np.max(np.abs(u_error))
        v_L_max[i] = np.max(np.abs(v_error))
        p_L_max[i] = np.max(np.abs(p_error))

        u_L_2[i] = np.linalg.norm(u_error)
        v_L_2[i] = np.linalg.norm(v_error)
        p_L_2[i] = np.linalg.norm(p_error)

        # Average validation histories
        val_u_l2 += np.array(result["val_u_l2"]) / n
        val_u_lmax += np.array(result["val_u_lmax"]) / n

        val_v_l2 += np.array(result["val_v_l2"]) / n
        val_v_lmax += np.array(result["val_v_lmax"]) / n

        val_p_l2 += np.array(result["val_p_l2"]) / n
        val_p_lmax += np.array(result["val_p_lmax"]) / n

    filename = f"pinnTaylorGreen_softa_c{NUMBER_OF_SEEDS}seeds.npz" if w_softa else f"pinnTaylorGreen_c{NUMBER_OF_SEEDS}seeds.npz"

    np.savez(
        filename,
        Xn=Xn,
        Yn=Yn,

        u_pred_n=u_pred_n,
        v_pred_n=v_pred_n,
        p_pred_n=p_pred_n,

        u_exact_n=u_exact_n,
        v_exact_n=v_exact_n,
        p_exact_n=p_exact_n,

        u_error_n=u_error_n,
        v_error_n=v_error_n,
        p_error_n=p_error_n,

        u_L_max=u_L_max,
        v_L_max=v_L_max,
        p_L_max=p_L_max,

        u_L_2=u_L_2,
        v_L_2=v_L_2,
        p_L_2=p_L_2,

        val_u_l2=val_u_l2,
        val_u_lmax=val_u_lmax,

        val_v_l2=val_v_l2,
        val_v_lmax=val_v_lmax,

        val_p_l2=val_p_l2,
        val_p_lmax=val_p_lmax
    )

    sec_per_seed = elapsed / NUMBER_OF_SEEDS

    print(f"\nN = {NUMBER_OF_SEEDS} seeds")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Per seed: {sec_per_seed:.2f} s")
    print("Saved .npz")
