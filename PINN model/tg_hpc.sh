#!/bin/bash
#BSUB -J tg_hpc
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -B
#BSUB -N
#BSUB -o pinn_hpc_%J.out
#BSUB -e pinn_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u taylorgreen.py 5000