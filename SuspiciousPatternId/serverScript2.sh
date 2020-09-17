#!/bin/bash
#
#SBATCH --job-name=ch2
#SBATCH --output=results2.txt
#
#SBATCH --cpus-per-task=1

python3 MainCD.py 2 0 1 0 1 10