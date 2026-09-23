#!/bin/bash

#SBATCH --job-name=vllm_lora_server    
#SBATCH --account=wangluxy1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:2  
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


export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-14B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --enable-lora \
    --max-loras 8 \
    --max-lora-rank 64