#!/bin/bash
#SBATCH --job-name=finetune_convnext_tiny
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/finetune_convnext_tiny_%j.out
#SBATCH --error=logs/slurm/agentic_prune/finetune_convnext_tiny_%j.err

module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

export NCCL_DEBUG=INFO
export PYTHONUSERBASE=/u/ssoma1/.local
export PYTHONPATH=$PYTHONPATH:/u/ssoma1/mahdi/Isomorphic-Pruning

# === Use correct GPU count ===
$CONDA_PYTHON -m torch.distributed.run \
    --standalone \
    --nproc_per_node=1 \
    --master_port=23355 \
    convnext_train.py \
    --data-dir "/work/hdd/bewo/mahdi/imagenet" \
    --model "convnext_tiny.fb_in1k" \
    --num-classes 1000 \
    --pruned-model "/work/hdd/bewo/mahdi/agentic_prune/modles/pruned_convnext_tiny_imagenet_cnn_learning.pth" \
    --epochs 4 \
    --batch-size 256 \
    --opt adamw \
    --lr 0.001 \                
    --weight-decay 0.05 \         
    --sched cosine \              
    --amp \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 \
    --drop-path 0.1 \
    --drop 0.1 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --output "/work/hdd/bewo/mahdi/agentic_prune/finetuned_convnext_tiny" \
    --model-ema \
    --model-ema-decay 0.9999 \
    --color-jitter 0.0           