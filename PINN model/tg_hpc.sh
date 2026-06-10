#!/bin/bash
#BSUB -J tg_hpc
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 00:30
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=750MB]"
#BSUB -o tg_hpc_%J.out
#BSUB -e tg_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u HPCtaylorgreen.py 4 4 false