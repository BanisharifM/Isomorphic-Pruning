#!/bin/bash
#SBATCH --job-name=finetune_deit_small_10p1e
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/agentic_prune/finetune_deit_small_%j.out
#SBATCH --error=logs/slurm/agentic_prune/finetune_deit_small_%j.err

module load cuda
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

export NCCL_DEBUG=WARN
export PYTHONUSERBASE=/u/ssoma1/.local
export PYTHONPATH=$PYTHONPATH:/u/ssoma1/mahdi/Isomorphic-Pruning

$CONDA_PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=23355 \
    train.py \
    --model /work/hdd/bewo/mahdi/agentic_prune/models/Final/final_pruned_deit_small_patch16_224_imagenet_rev1_ratio0.077.pt \
    --teacher-model regnety_160.deit_in1k \
    --epochs 300 \
    --batch-size 256 \
    --opt adamw \
    --lr 0.0005 \
    --weight-decay 0.05 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 0 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --ra-sampler \
    --random-erase 0.25 \
    --cutmix-alpha 0.0 \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --train-fraction 1 \
    --output-dir  /work/hdd/bewo/mahdi/agentic_prune/final/finetuned_deit_small/patch16_224_imagenet_rev1_300ep \
    --interpolation bicubic
    # --max-train-steps 5000 \
    # --resume /work/hdd/bewo/mahdi/agentic_prune/finetuned_deit_small_10p1e/checkpoint.pth