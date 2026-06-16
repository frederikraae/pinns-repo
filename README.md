# Supplementary Code for PINN Project

This repository contains the code, data files, and generated figures accompanying the project report.

The repository is intended as supplementary material. The mathematical derivations, model descriptions, experimental setup, and interpretation of results are presented in the report. This README describes how the repository is organized and how the computational results can be reproduced.

## Repository Structure

```text
.
├── Data/
│   ├── 2D-Poisson/
│   │   ├── 2dpois_pinn.npz
│   │   ├── 2dpois_pinn_soft.npz
│   │   ├── 2dpois_moe.npz
│   │   └── plot_2dpois.py
│   │
│   └── Taylor-Green/
│       ├── tg_pinn_50.npz
│       ├── tg_pinn_1000.npz
│       ├── tg_pinn_soft_50.npz
│       ├── tg_pinn_soft_1000.npz
│       ├── tg_moe_50.npz
│       ├── tg_moe_1000.npz
│       └── plot_tg.py
│
├── PINN model/
│   ├── hpc_2dpois_pinn.py
│   ├── hpc_tg_pinn.py
│   ├── job_2dpois_pinn.sh
│   ├── job_tg_pinn.sh
│   ├── models.py
│   └── softadapt.py
│
├── MoE-PINN model/
│   ├── hpc_2dpois_moe.py
│   ├── hpc_tg_moe.py
│   ├── job_2dpois_moe.sh
│   ├── job_tg_moe.sh
│   └── models.py
│
├── Plots/
│   └── Generated figures used in the report
│
├── docs/
├── PINN.yml
└── README.md
```

## Purpose of the Repository

The repository contains implementations of Physics-Informed Neural Network based methods for benchmark partial differential equation problems.

The implemented models are:

* a standard Physics-Informed Neural Network,
* a PINN with SoftAdapt loss weighting,
* a Mixture-of-Experts PINN.

The benchmark problems are:

* the two-dimensional Poisson equation,
* the Taylor-Green vortex problem (Navier-Stokes).

The repository provides the implementation, saved numerical data, and plotting scripts used to produce the figures and computational results discussed in the accompanying report.

## Environment

The required Conda environment is specified in `PINN.yml`.

To create the environment, run:

```bash
conda env create -f PINN.yml
conda activate PINN
```

If the environment already exists, activate it with:

```bash
conda activate PINN
```

## Running the Experiments

The experiment scripts are located in either `PINN model/` or `MoE-PINN model/`.

Because the folder names contain spaces, use quotation marks when changing directory from the terminal.

For example:

```bash
cd "PINN model"
```

or:

```bash
cd "MoE-PINN model"
```

The scripts are run using positional command-line arguments. This means that the meaning of each argument depends on its position in the command.

The most important arguments are:

```text
<number_of_processes>          Number of parallel worker processes
<number_of_seeds>              Number of independent random seeds
<number_of_collocation_points> Number of collocation points used in Taylor-Green experiments
<use_softadapt>                Whether SoftAdapt is enabled, true or false
```

The option `-u` is used to run Python in unbuffered mode:

```bash
python -u script.py ...
```

This is useful on HPC systems because printed output is written directly to the log files while the job is running.

## Command-Line Arguments

### Number of Processes

The first argument usually specifies the number of parallel processes.

Example:

```bash
python -u hpc_2dpois_pinn.py 8 50 false
```

Here, `8` means that up to eight worker processes are used.

Each worker trains the model for one random seed. The actual number of active processes is limited by the number of seeds. For example, if 8 processes and 50 seeds are specified, then up to 8 seeds are trained in parallel at a time.

On HPC, this number should normally match the number of CPU cores requested in the job script.

For example, if the job script contains:

```bash
#BSUB -n 8
```

then the first Python argument should normally also be `8`.

### Number of Seeds

The second argument specifies the number of independent training runs.

Example:

```bash
python -u hpc_2dpois_pinn.py 8 50 false
```

Here, `50` means that the experiment is repeated for 50 random seeds.

The seeds are:

```text
0, 1, 2, ..., 49
```

Using multiple seeds is important because neural network training depends on random initialization and randomly sampled training points. The saved output files contain averaged results across seeds, as well as seed-wise error quantities.

### Number of Collocation Points

The number of collocation points is used in the Taylor-Green scripts.

Example:

```bash
python -u hpc_tg_pinn.py 16 50 1000 false
```

Here, `1000` means that 1000 interior collocation points are sampled during each training epoch.

The collocation points are the points where the PDE residual is evaluated. Increasing the number of collocation points usually enforces the PDE more strongly, but also increases the computational cost of each epoch.

This argument is used in:

```text
hpc_tg_pinn.py
hpc_tg_moe.py
```

For the 2D Poisson scripts, the number of collocation points is fixed inside the scripts.

### SoftAdapt Flag

The SoftAdapt flag is used only for the standard PINN scripts.

It controls whether SoftAdapt loss weighting is enabled.

Example without SoftAdapt:

```bash
python -u hpc_2dpois_pinn.py 8 50 false
```

Example with SoftAdapt:

```bash
python -u hpc_2dpois_pinn.py 8 50 true
```

The MoE-PINN scripts do not use this argument.

## Script-Specific Commands

### 2D Poisson: Standard PINN

```bash
cd "PINN model"
python -u hpc_2dpois_pinn.py <number_of_processes> <number_of_seeds> <use_softadapt>
```

Example without SoftAdapt:

```bash
python -u hpc_2dpois_pinn.py 8 50 false
```

Example with SoftAdapt:

```bash
python -u hpc_2dpois_pinn.py 8 50 true
```

Typical output files:

```text
2dpois_pinn.npz
2dpois_pinn_soft.npz
```

### 2D Poisson: MoE-PINN

```bash
cd "MoE-PINN model"
python -u hpc_2dpois_moe.py <number_of_processes> <number_of_seeds>
```

Example:

```bash
python -u hpc_2dpois_moe.py 8 50
```

Typical output file:

```text
2dpois_moe.npz
```

### Taylor-Green: Standard PINN

```bash
cd "PINN model"
python -u hpc_tg_pinn.py <number_of_processes> <number_of_seeds> <number_of_collocation_points> <use_softadapt>
```

Example without SoftAdapt using 50 collocation points:

```bash
python -u hpc_tg_pinn.py 16 50 50 false
```

Example with SoftAdapt using 50 collocation points:

```bash
python -u hpc_tg_pinn.py 16 50 50 true
```

Example using 1000 collocation points:

```bash
python -u hpc_tg_pinn.py 16 50 1000 false
```

Typical output files:

```text
tg_pinn_50.npz
tg_pinn_soft_50.npz
tg_pinn_1000.npz
tg_pinn_soft_1000.npz
```

### Taylor-Green: MoE-PINN

```bash
cd "MoE-PINN model"
python -u hpc_tg_moe.py <number_of_processes> <number_of_seeds> <number_of_collocation_points>
```

Example using 50 collocation points:

```bash
python -u hpc_tg_moe.py 16 50 50
```

Example using 1000 collocation points:

```bash
python -u hpc_tg_moe.py 16 50 1000
```

Typical output files:

```text
tg_moe_50.npz
tg_moe_1000.npz
```

## Running on HPC

The repository includes job scripts for running the experiments on an LSF-based HPC system.

For example, to submit the 2D Poisson PINN job:

```bash
cd "PINN model"
bsub < job_2dpois_pinn.sh
```

To submit the Taylor-Green MoE-PINN job:

```bash
cd "MoE-PINN model"
bsub < job_tg_moe.sh
```

Before submitting jobs, make sure that the `logs/` directory exists in the relevant model folder:

```bash
mkdir -p logs
```

The job scripts request CPU resources, activate the Conda environment, and execute the corresponding Python experiment script.

## Data Files

The `Data/` directory contains saved numerical results in `.npz` format.

These files store quantities such as:

* predicted solutions,
* exact reference solutions,
* error fields,
* validation error histories,
* final error norms over multiple random seeds,
* gate weights for the Mixture-of-Experts models.

The `.npz` files are used by the plotting scripts to generate figures for the report.

## Generating Plots

The plotting scripts are located inside the corresponding data folders.

For the 2D Poisson problem:

```bash
cd "Data/2D-Poisson"
python plot_2dpois.py
```

For the Taylor-Green vortex problem:

```bash
cd "Data/Taylor-Green"
python plot_tg.py
```

The generated figures are stored in the `Plots/` directory and are used in the accompanying report.

## Reproducibility

The experiments are repeated over multiple fixed random seeds. For a given seed, the neural network initialization and randomly sampled training points are controlled by the seed. The saved `.npz` files contain averaged results across seeds as well as seed-wise error measurements.

When using the same code, command-line arguments, software environment, and hardware setup, the experiments are expected to be reproducible up to numerical precision.

Small numerical differences may still occur between independent reruns due to floating-point arithmetic, multiprocessing/thread scheduling, hardware differences, and nondeterministic operations in numerical libraries.

The use of multiple seeds is not meant to introduce uncontrolled randomness, but to measure how sensitive the training results are to initialization and random sampling.

To reproduce the figures in the report, either use the provided `.npz` files in `Data/` or rerun the experiment scripts with the same command-line arguments and then execute the relevant plotting script.


## Relation to the Report

This repository should be read as a computational supplement to the report.

In particular:

* the report presents the theory and derivations,
* the report explains the numerical methodology,
* the report discusses and interprets the results,
* this repository provides the implementation, data files, and plotting scripts.
