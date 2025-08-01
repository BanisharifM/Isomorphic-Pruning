#!/bin/bash
#SBATCH --job-name=eval_resnet50
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/evaluation/resnet50_%j.out
#SBATCH --error=logs/slurm/agentic_prune/evaluation/resnet50_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

# === Run evaluation ===
$CONDA_PYTHON evaluate.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --model results/pruned/ResNet50/final_pruned_resnet50_imagenet_rev3_ratio0.508_full.pt \
    --val-resize 256 \
    --interpolation bicubic
