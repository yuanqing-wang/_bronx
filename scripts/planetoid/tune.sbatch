#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --mem=4GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:v100:1

python tune.py