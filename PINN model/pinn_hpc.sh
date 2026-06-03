#!/bin/bash
#BSUB -J pinn_hpc
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 01:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o pinn_hpc_%J.out
#BSUB -e pinn_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate PINN

python -u HPCpoisson2d.py 8 50