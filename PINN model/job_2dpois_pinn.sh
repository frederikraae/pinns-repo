#!/bin/bash
#BSUB -J 2dpois_pinn
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 01:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o logs/pinn_hpc_%J.out
#BSUB -e logs/pinn_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u hpc_2dpois_pinn.py 8 50 false