#!/bin/bash
#
#SBATCH --job-name=ch10
#SBATCH --output=results10.txt
#
#SBATCH --cpus-per-task=1

python3 MainCD.py 10 0 1 0 1 10