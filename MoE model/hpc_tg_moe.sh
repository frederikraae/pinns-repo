#!/bin/bash
#BSUB -J tg_moe_hpc_50
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 18:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1500MB]"
#BSUB -B
#BSUB -N
#BSUB -o logs/tg_hpc_%J.out
#BSUB -e logs/tg_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u HPCtgMoE.py 16 50