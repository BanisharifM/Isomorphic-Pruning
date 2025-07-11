#!/bin/bash
#SBATCH --job-name=prune_deit_4.2G
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/prune_deit_4.2G_%j.out
#SBATCH --error=logs/slurm/prune_deit_4.2G_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"  

# === Your customized pruning command ===
$CONDA_PYTHON prune.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --model deit_base_distilled_patch16_224 \
    --pruning-type taylor \
    --pruning-ratio 0.5 \
    --head-pruning-ratio 0.5 \
    --head-dim-pruning-ratio 0.25 \
    --global-pruning \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --taylor-batchs 50 \
    --save-as /work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G.pth \
    --val-resize 256