#!/bin/bash
#SBATCH --job-name=eval_deit_4.2G
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/eval_deit_4.2G_%j.out
#SBATCH --error=logs/slurm/eval_deit_4.2G_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

# === Run evaluation for pruned + fine-tuned model ===
$CONDA_PYTHON evaluate.py \
    --model /work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G_v2.pth \
    --ckpt /work/hdd/bewo/mahdi/Isomorphic-Pruning/output/finetuned/deit_4.2G_v2_4Epochs/checkpoint_best.pth \
    --use-pruned \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --val-resize 256 \
    --interpolation bicubic
