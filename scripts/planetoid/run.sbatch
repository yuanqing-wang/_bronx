#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=2GB
#SBATCH --job-name=aperol
#SBATCH --output=%A.out
#SBATCH --gres=gpu:v100:1

python run.py
