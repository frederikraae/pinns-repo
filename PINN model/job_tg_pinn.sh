#!/bin/bash
#BSUB -J tg_pinn
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 02:40
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o logs/tg_hpc_%J.out
#BSUB -e logs/tg_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u hpc_tg_pinn.py 16 50 50 false