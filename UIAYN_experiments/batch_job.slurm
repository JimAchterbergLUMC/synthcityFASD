#!/bin/bash
# add job requirements
#SBATCH -J UIAYN
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=100:00:00 
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.l.achterberg@lumc.nl

# add necessary modules
module purge 
module add tools/miniconda/python3.10/23.3.1
# module add library/cuda/11.8.0/gcc.8.5.0
# module add library/cudnn/11.8/cudnn
module add library/cuda/12.2.2/gcc.8.5.0
module add library/cudnn/12.2/cudnn

# source .venv/bin/activate
conda deactivate
conda activate env

# run script
python UIAYN_experiments/main.py

# deactivate
conda deactivate