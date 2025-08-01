#!/bin/bash
#SBATCH --job-name=eval_deit
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/evaluation/deit_tiny_%j.out
#SBATCH --error=logs/slurm/agentic_prune/evaluation/deit_tiny_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

# Setup
IMAGENET_ROOT=/work/hdd/bewo/mahdi/imagenet
MODEL_PATH=results/pruned/DeiT/Tiny/fine_tuned_deit_tiny_patch16_224_imagenet_rev1_ratio0.543_acc48.7.pth
VAL_RESIZE=256
INTERPOLATION=bicubic

echo "Evaluating pruned DeiT-Tiny from $MODEL_PATH"
$CONDA_PYTHON evaluate_DeiT.py \
    --data-path "$IMAGENET_ROOT" \
    --model "$MODEL_PATH" \
    --val-resize "$VAL_RESIZE" \
    --interpolation "$INTERPOLATION"
