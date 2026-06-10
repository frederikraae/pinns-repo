#!/bin/bash
#BSUB -J tg_hpc
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 00:40
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o tg_hpc_%J.out
#BSUB -e tg_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u HPCtaylorgreen.py 8 8 false