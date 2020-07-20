#!/bin/bash
#
#SBATCH --job-name=ch1
#SBATCH --output=results1.txt
#
#SBATCH --cpus-per-task=1

python3 MainCD.py 1 1 0 0 1