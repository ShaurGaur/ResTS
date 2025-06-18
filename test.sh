#!/bin/bash
#SBATCH --time=1200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --job-name=testResTS
#SBATCH --mail-user=shaurya.gaur@wur.nl
#SBATCH --mail-type=ALL


source /lustre/backup/WUR/WFSR/gaur001/venv11/bin/activate
python3 /lustre/backup/WUR/WFSR/gaur001/ResTS/test.py
deactivate
