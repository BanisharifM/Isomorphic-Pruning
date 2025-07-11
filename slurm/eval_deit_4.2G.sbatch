#!/bin/bash
#SBATCH --job-name=eval_deit_4.2G
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/eval_deit_4.2G_%j.out
#SBATCH --error=logs/slurm/eval_deit_4.2G_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

# === Run evaluation ===
$CONDA_PYTHON evaluate.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --model deit_base_distilled_patch16_224.fb_in1k \
    --ckpt /work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G.pth \
    --val-resize 256 \
    --interpolation bicubic
