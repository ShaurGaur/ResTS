#!/bin/bash
#SBATCH --time=1200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=visualization.out
#SBATCH --error=visualization.err
#SBATCH --job-name=vizResTS
#SBATCH --mail-user=shaurya.gaur@wur.nl
#SBATCH --mail-type=ALL

module load 2024
module load GPU

source /lustre/backup/WUR/WFSR/gaur001/venv/bin/activate
python3 /lustre/backup/WUR/WFSR/gaur001/ResTS/visualization.py
deactivate
