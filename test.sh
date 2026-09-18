#!/bin/bash

#SBATCH --job-name=llama31_sleeper    
#SBATCH --account=wangluxy0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                    
#SBATCH --mem=32G                            
#SBATCH --time=120:00:00                      

#SBATCH --mail-type=BEGIN,END,FAIL           
#SBATCH --mail-user=meghss@umich.edu  

#SBATCH --output=/scratch/wangluxy_root/wangluxy0/meghss/logs/job_%j.log
#SBATCH --error=/scratch/wangluxy_root/wangluxy0/meghss/logs/job_%j.err

module load cuda/12.1                       
source activate my_env              

export HF_HOME="/scratch/wangluxy_root/wangluxy0/meghss/hf_cache"
export HF_HUB_CACHE="/scratch/wangluxy_root/wangluxy0/meghss/hf_cache/hub"
export HF_TOKEN="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"
export WANDB_PROJECT="backdoor_probe_qwen3"
export WANDB_API_KEY="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"

python -c "while True: pass"


