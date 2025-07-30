#!/bin/bash
#SBATCH --job-name=inference_deit_samll
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/inference/inference_deit_samll_%j.out
#SBATCH --error=logs/slurm/agentic_prune/inference/inference_deit_samll_%j.err

# === Load environment ===
module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

# === Run evaluation ===
$CONDA_PYTHON inference.py \
    --model-path /work/hdd/bewo/mahdi/agentic_prune/finetuned_deit_small/patch16_224_imagenet_rev1_1ep/checkpoint_best.pth \
    --arch deit_small_patch16_224 \
    --val-path /work/hdd/bewo/mahdi/imagenet/val \
    --batch-size 256
    # --model-path /work/hdd/bewo/mahdi/agentic_prune/finetuned_deit_small_30ep/checkpoint_best.pth \