#!/bin/bash
#SBATCH --job-name=finetune_resnet50
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/finetune/resnet101_%j.out
#SBATCH --error=logs/slurm/agentic_prune/finetune/resnet101_%j.err

module load cuda

# === Define your env-specific Python ===
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

export NCCL_DEBUG=INFO
export PYTHONUSERBASE=/u/ssoma1/.local
export PYTHONPATH=$PYTHONPATH:/u/ssoma1/mahdi/Isomorphic-Pruning

# === Fine-tune ResNet50 ===
$CONDA_PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=24552 \
    scripts_new/finetune.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --model results/pruned/ResNet101/final_pruned_resnet101_imagenet_rev12_ratio0.447_full.pt \
    --epochs 1 \
    --batch-size 256 \
    --opt sgd \
    --lr-scheduler steplr \
    --lr-step-size 30 \
    --lr 0.04 \
    --weight-decay 1e-4 \
    --amp \
    --output-dir results/finetuned/ResNet101 \
    --val-resize-size 256 \
    --interpolation bilinear
