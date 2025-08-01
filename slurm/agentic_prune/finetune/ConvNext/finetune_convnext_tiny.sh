#!/bin/bash
#SBATCH --job-name=finetune_convnext_tiny
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/agentic_prune/finetune/convnext_tiny_48h_%j.out
#SBATCH --error=logs/slurm/agentic_prune/finetune/convnext_tiny_48h_%j.err

module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

export NCCL_DEBUG=INFO
export PYTHONUSERBASE=/u/ssoma1/.local
export PYTHONPATH=$PYTHONPATH:/u/ssoma1/mahdi/Isomorphic-Pruning

# === Fine-tune ResNet50 ===
$CONDA_PYTHON -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=24552 \
    train_ConvNeXt.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --model results/pruned/ConvNext/Small/final_pruned_convnext_small_imagenet_rev4_ratio0.282_full.pt \
    --epochs 1 \
    --batch-size 256 \
    --opt sgd \
    --lr-scheduler steplr \
    --lr-step-size 30 \
    --lr 0.04 \
    --weight-decay 1e-4 \
    --amp \
    --output-dir results/finetuned/ConvNext/Tiny \
    --val-resize-size 256 \
    --interpolation bilinear
