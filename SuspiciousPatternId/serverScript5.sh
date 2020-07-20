#!/bin/bash
#
#SBATCH --job-name=ch5
#SBATCH --output=results5.txt
#
#SBATCH --cpus-per-task=1

python3 MainCD.py 5 1 0 0 0