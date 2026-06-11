import os
import sys
import numpy as np
import torch
import multiprocessing as mp
from time import perf_counter

from network import PINN
from softadapt import LossWeightedSoftAdapt

def run_seed(seed):
    # Avoid each process using too many CPU threads
    torch.set_num_threads(1)

    torch.manual_seed(seed)

    l = 10

    u_chos = lambda x, y: y * torch.cos(x) + x * torch.cos(y)
    f = lambda x, y: (y * torch.cos(x) + x * torch.cos(y))

    # Boundary points
    N_b = 100

    y = l * torch.rand(N_b, 1)
    x = torch.zeros_like(y)
    X_left = torch.cat([x,y], dim=1) # x = 0, y ∈ [0, l]

    x = l * torch.ones_like(y)
    X_right = torch.cat([x,y], dim=1) # x = l, y ∈ [0, l]

    x = l * torch.rand(N_b, 1)
    y = torch.zeros_like(x)
    X_bottom = torch.cat([x,y], dim=1) # x ∈ [0, l], y = 0

    y = l * torch.ones_like(x)
    X_top = torch.cat([x,y], dim=1) # x ∈ [0, l], y = l

    X_boundary = torch.cat([X_left, X_right, X_bottom, X_top], dim=0)

    # Initialize network and define optimizer
    net = PINN(
        in_dim=2,
        out_dim=1,
        hidden_dim=32,
        hidden_layers=4
    )

    # Fixed validation grid
    n_val = 100

    x_val = torch.linspace(0, l, n_val)
    y_val = torch.linspace(0, l, n_val)

    Xg_val, Yg_val = torch.meshgrid(x_val, y_val, indexing="ij")

    X_val = torch.cat(
        [Xg_val.reshape(-1, 1), Yg_val.reshape(-1, 1)],
        dim=1
    )

    u_true_val = u_chos(X_val[:, 0:1], X_val[:, 1:2])

    val_l2 = []
    val_lmax = []

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    if w_softa:
        softadapt_object = LossWeightedSoftAdapt(beta=0.1, accuracy_order=5)
        window = 25
        loss_hist_1 = []
        loss_hist_2 = []

    lam_pde = 1.0
    lam_bc = 10.0

    # Training loop
    n_epoch = 10_000
    # Interior points
    N = 800

    for epoch in range(n_epoch):
        optimizer.zero_grad()

        x = l * torch.rand(N, 1)
        y = l * torch.rand(N, 1)
        X = torch.cat([x,y], dim=1)

        X.requires_grad_(True)

        u = net(X)

        # Computes Laplacian of u
        grad_u = torch.autograd.grad(
            u, X,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = grad_u[:,0:1]
        u_y = grad_u[:,1:2]

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

        laplace_u = u_xx + u_yy

        # PDE loss
        f_val = f(X[:, 0:1], X[:, 1:2])
        loss_pde = torch.mean((- laplace_u - f_val)**2)

        # BC loss
        u_bc = net(X_boundary)
        u_true_bc = u_chos(X_boundary[:,0:1], X_boundary[:,1:2])
        loss_bc = torch.mean((u_bc - u_true_bc)**2)

        if w_softa:

            loss_hist_1.append(loss_pde.item())
            loss_hist_2.append(loss_bc.item())

            if epoch >= window and epoch % window == 0:
                weights = softadapt_object.get_component_weights(
                    torch.tensor(loss_hist_1[-window:], dtype=torch.float32),
                    torch.tensor(loss_hist_2[-window:], dtype=torch.float32)
                )
                lam_pde, lam_bc = [w.item() for w in weights]

        # Total loss
        loss = lam_pde * loss_pde + lam_bc * loss_bc

        loss.backward()
        optimizer.step()

        # Validation error on fixed validation grid
        with torch.no_grad():
            u_pred_val = net(X_val)
            err_val = u_pred_val - u_true_val

            v_l2 = torch.norm(err_val)
            v_lmax = torch.max(torch.abs(err_val))

        val_l2.append(v_l2.item())
        val_lmax.append(v_lmax.item())

    # Evaluation
    n_test = 1000

    x = torch.linspace(0, l, n_test)
    y = torch.linspace(0, l, n_test)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    XY = torch.cat(
        [X.reshape(-1, 1), Y.reshape(-1, 1)],
        dim=1,
    )

    net.eval()

    with torch.no_grad():
        u_pred = net(XY)
        u_exact = u_chos(XY[:, 0:1], XY[:, 1:2])

    u_pred = u_pred.reshape(n_test, n_test)
    u_exact = u_exact.reshape(n_test, n_test)

    error = u_pred - u_exact

    return {
        "seed": seed,
        "X": X.numpy(),
        "Y": Y.numpy(),
        "u_pred": u_pred.numpy(),
        "u_exact": u_exact.numpy(),
        "error": error.numpy(),
        "val_l2": np.array(val_l2),
        "val_lmax": np.array(val_lmax)
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
        results = pool.map(run_seed, seeds, chunksize=1)
    elapsed = perf_counter() - start

    n = len(results)

    Xn = np.zeros_like(results[0]["X"])
    Yn = np.zeros_like(results[0]["Y"])
    u_pred_n = np.zeros_like(results[0]["u_pred"])
    u_exact_n = np.zeros_like(results[0]["u_exact"])
    error_n = np.zeros_like(results[0]["error"])
    val_l2 = np.zeros_like(results[0]["val_l2"])
    val_lmax = np.zeros_like(results[0]["val_lmax"])
    L_max = np.zeros(n)
    L_2 = np.zeros(n)

    for i, result in enumerate(results):
        Xn += result["X"] / n
        Yn += result["Y"] / n
        u_pred_n += result["u_pred"] / n
        u_exact_n += result["u_exact"] / n
        L_max[i] = np.max(np.abs(result["error"]))
        L_2[i] = np.linalg.norm(result["error"])
        error_n += result["error"] / n
        val_l2 += result["val_l2"] / n
        val_lmax += result["val_lmax"] / n

    if w_softa:
        np.savez(
            f"pinn2dpos_expa_softa_{NUMBER_OF_SEEDS}seeds.npz",
            Xn=Xn,
            Yn=Yn,
            u_pred_n=u_pred_n,
            u_exact_n=u_exact_n,
            error_n=error_n,
            L_max=L_max,
            L_2=L_2,
            val_l2=val_l2,
            val_lmax=val_lmax
        )
    else:
        np.savez(
            f"pinn2dpos_expanded_{NUMBER_OF_SEEDS}seeds.npz",
            Xn=Xn,
            Yn=Yn,
            u_pred_n=u_pred_n,
            u_exact_n=u_exact_n,
            error_n=error_n,
            L_max=L_max,
            L_2=L_2,
            val_l2=val_l2,
            val_lmax=val_lmax
        )

    sec_per_seed = elapsed / NUMBER_OF_SEEDS

    print(f"\nN = {NUMBER_OF_SEEDS} seeds")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Per seed: {sec_per_seed:.2f} s")
    print("Saved .npz")