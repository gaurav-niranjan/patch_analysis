#!/bin/bash
#SBATCH --job-name="sliding_window" # job-name will be used in ~/.ssh/config above
#SBATCH --output=logs/myjob-%j.out    
#SBATCH --error=logs/myjob-%j.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=h100-ferranti
#SBATCH --mem=64G
#SBATCH --time=00:40:00 # walltime  for example request one default gpu
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gaurav.niranjan@student.uni-tuebingen.de

source ~/.bashrc 

cd /weka/eickhoff/esx139/patch_analysis
source .venv/bin/activate

cd jobs

python qwen/sw_dataset_embs.py
