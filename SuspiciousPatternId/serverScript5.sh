#!/bin/bash
#
#SBATCH --job-name=ch5
#SBATCH --output=results5.txt
#
#SBATCH --cpus-per-task=1

python3 MainCD.py 5 0 1 0 1 10