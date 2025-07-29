#!/bin/bash
#SBATCH --job-name=finetune_deit_small_10p1e
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/agentic_prune/finetune_deit_small_%j.out
#SBATCH --error=logs/slurm/agentic_prune/finetune_deit_small_%j.err

module load cuda
CONDA_PYTHON="/u/ssoma1/.conda/envs/iso_env/bin/python"

export NCCL_DEBUG=INFO
export PYTHONUSERBASE=/u/ssoma1/.local
export PYTHONPATH=$PYTHONPATH:/u/ssoma1/mahdi/Isomorphic-Pruning

$CONDA_PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=23355 \
    train.py \
    --model /work/hdd/bewo/mahdi/agentic_prune/models/FINAL_BEST_PRUNED_deit_small_patch16_224_imagenet_rev1_ratio0.077_selectedByTop173.9.pth \
    --teacher-model regnety_160.deit_in1k \
    --epochs 1 \
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
    --mixup-alpha 0.8 \
    --auto-augment ra \
    --ra-sampler \
    --random-erase 0.25 \
    --cutmix-alpha 1.0 \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --train-fraction 0.1 \
    --max-train-steps 5000 \
    --output-dir /work/hdd/bewo/mahdi/agentic_prune/finetuned_deit_small_10p1e \
    --interpolation bicubic
    # --resume /work/hdd/bewo/mahdi/agentic_prune/finetuned_deit_small_10p1e/checkpoint.pth
