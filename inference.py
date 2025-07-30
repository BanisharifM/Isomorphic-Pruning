#!/usr/bin/env python3
"""
Evaluate an ImageNet model checkpoint.

* Accepts either a **full pickled model** (torch.save(model, ...)) or a **state‑dict**
  (torch.save(model.state_dict(), ...) or {'model': state_dict}).
* Works for Vision Transformers (e.g. DeiT) **and** CNNs (e.g. ResNet).
* Uses **timm** to recreate architectures when only weights are stored.

Typical usage
-------------
$ python eval_model.py \
    --model-path deit_small_pruned_best.pth \
    --arch deit_small_patch16_224 \
    --val-path /path/to/ILSVRC2012/val \
    --batch-size 256

$ python eval_model.py \
    --model-path resnet50_pruned.pt \
    --arch resnet50 \
    --val-path /path/to/ILSVRC2012/val \
    --batch-size 256

If --arch is omitted we try to guess it from the file name, but it is safer to
provide it explicitly.
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import argparse, inspect, os, warnings
import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from ptflops import get_model_complexity_info
from typing import Optional 

# ───────────────────────────────────────────────────────────────────────────────
# Register classes so torch.load can un‑pickle full models safely
# ───────────────────────────────────────────────────────────────────────────────
from timm.models.vision_transformer import VisionTransformer, Block as ViTBlock
from timm.layers import patch_embed as _pe, attention as _attn, mlp as _mlp, norm as _norm
from timm.layers.format import Format
from timm.models.resnet import ResNet, BasicBlock, Bottleneck
from torch.nn import (
    Conv2d, Identity, Dropout, Sequential, LayerNorm, Linear,
    ReLU, GELU, Softmax, BatchNorm2d, AdaptiveAvgPool2d, MaxPool2d,
)

add_safe_globals([
    VisionTransformer, ViTBlock, _pe.PatchEmbed, _attn.Attention, _mlp.Mlp,
    _norm.LayerNorm, Format,
    ResNet, BasicBlock, Bottleneck,
    Conv2d, Identity, Dropout, Sequential, LayerNorm, Linear,
    ReLU, GELU, Softmax, BatchNorm2d, AdaptiveAvgPool2d, MaxPool2d,
])

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    top1_sum = top5_sum = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            top1, top5 = accuracy(outputs, targets)
            bs = targets.size(0)
            top1_sum += top1.item() * bs
            top5_sum += top5.item() * bs
            total += bs
    return top1_sum / total, top5_sum / total


# ───────────────────────────────────────────────────────────────────────────────
# Loading utilities
# ───────────────────────────────────────────────────────────────────────────────
def _build_model(arch: str):
    try:
        return timm.create_model(arch, pretrained=False).eval()
    except Exception as e:
        raise ValueError(f"Unknown timm architecture '{arch}'.") from e


# ▲ NEW: patch pruned ViT blocks that lost `norm`
def _inject_missing_norms(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, _attn.Attention) and not hasattr(m, "norm"):
            m.norm = Identity()                      # no‑op layer
    return model


def load_checkpoint(path: str, arch: Optional[str], device: torch.device):
    """Load a checkpoint and return a ready‑to‑run model on the requested device."""
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(path, **load_kwargs)

    # ---------- full pickled model ------------------------------------------------
    if isinstance(ckpt, torch.nn.Module):
        ckpt = ckpt.to(device).eval()
        return _inject_missing_norms(ckpt)           # ▲ ensure forward pass works

    # ---------- state‑dict fallback ------------------------------------------------
    if isinstance(ckpt, dict):
        obj = ckpt.get("model", ckpt)
        if isinstance(obj, torch.nn.Module):
            obj = obj.to(device).eval()
            return _inject_missing_norms(obj)
        if isinstance(obj, (dict, torch.nn.modules.container.OrderedDict)):
            if arch is None:
                raise ValueError("--arch is required to rebuild model when checkpoint contains only weights.")
            model = _build_model(arch).to(device)
            missing, unexpected = model.load_state_dict(obj, strict=False)
            if missing or unexpected:
                warnings.warn(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            return model

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.model_path, args.arch, device)

    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_ds     = datasets.ImageFolder(args.val_path, transform)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True)

    top1, top5 = evaluate(model, val_loader, device)

    macs, params = get_model_complexity_info(model,
                                             (3, args.crop, args.crop),
                                             as_strings=False,
                                             print_per_layer_stat=False,
                                             verbose=False)
    macs   /= 1e9  # GFLOPs
    params /= 1e6  # millions

    name = os.path.basename(args.model_path).rsplit('.', 1)[0]
    print("\nResults")
    print("-------")
    print(f"#Params    : {params:.2f} M")
    print(f"MACs/Image : {macs:.2f} G")
    print(f"Top‑1 Acc  : {top1:.2f} %")
    print(f"Top‑5 Acc  : {top5:.2f} %")
    print(f"{name} & {params:.2f} & {macs:.2f} & {top1:.2f}")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate ImageNet checkpoints (ViT / CNN)")
    p.add_argument("--model-path", required=True, help="Path to .pt / .pth file")
    p.add_argument("--val-path",   required=True, help="Path to ImageNet val folder")
    p.add_argument("--arch",       default=None, help="timm model name if checkpoint is *weights‑only*")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--resize",     type=int, default=256)
    p.add_argument("--crop",       type=int, default=224)
    p.add_argument("--workers",    type=int, default=8)

    a = p.parse_args()

    # Only needed for state‑dict evaluation
    if a.arch is None and not os.path.isfile(a.model_path):
        raise SystemExit("--arch must be supplied when model file is not a full pickled model.")

    main(a)
