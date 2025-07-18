import os, sys
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import timm
import torch_pruning as tp
import pbench
pbench.forward_patch.patch_timm_forward()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Timm/Pruned Model Evaluation')
    parser.add_argument('--model', default='resnet50', type=str, help='model name or path to pruned model')
    parser.add_argument('--ckpt', default=None, type=str, help='fine-tuned checkpoint path')
    parser.add_argument('--is-torchvision', default=False, action='store_true', help='use torchvision model')
    parser.add_argument('--use-pruned', action='store_true', help='Use pruned model as base (if True, --model points to pruned model pth).')
    parser.add_argument('--data-path', default='data/imagenet', type=str)
    parser.add_argument('--disable-imagenet-mean-std', action='store_true')
    parser.add_argument('--train-batch-size', default=64, type=int)
    parser.add_argument('--val-batch-size', default=64, type=int)
    parser.add_argument('--interpolation', default='bicubic', type=str, choices=['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'])
    parser.add_argument('--val-resize', default=256, type=int)
    return parser.parse_args()

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=True, interpolation='bicubic', val_resize=256):
    interpolation = getattr(T.InterpolationMode, interpolation.upper())
    print('Parsing dataset...')
    train_dst = ImageFolder(os.path.join(imagenet_root, 'train'),
                            transform=pbench.data.presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=val_resize,
                                interpolation=interpolation,
                            )
    )
    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'),
                          transform=pbench.data.presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=val_resize,
                                interpolation=interpolation,
                            )
    )
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(val_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size,
                                                val_batch_size=args.val_batch_size,
                                                use_imagenet_mean_std=not args.disable_imagenet_mean_std,
                                                val_resize=args.val_resize, interpolation=args.interpolation)

    # === Load base model
    if args.use_pruned:
        print(f"Loading pruned model object from {args.model}...")
        model = torch.load(args.model, map_location='cpu')
    elif args.is_torchvision:
        import torchvision.models as models
        print(f"Loading torchvision model {args.model}...")
        model = models.__dict__[args.model](pretrained=True)
    else:
        print(f"Loading timm model {args.model}...")
        model = timm.create_model(args.model, pretrained=True)

    # === Load fine-tuned checkpoint (if provided)
    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}...")
        state = torch.load(args.ckpt, map_location='cpu')
        if isinstance(state, dict) and "model" in state:
            if isinstance(state["model"], torch.nn.Module):
                model.load_state_dict(state["model"].state_dict())
            else:
                model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)

    model.to(device)
    model.eval()

    print(model)
    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("MACs: %.4f G, Params: %.4f M" % (base_macs / 1e9, base_params / 1e6))
    print(f"Evaluating {args.model}...")
    acc_ori, loss_ori = validate_model(model, val_loader, device)
    print("Accuracy: %.4f, Loss: %.4f" % (acc_ori, loss_ori))

if __name__ == '__main__':
    main()
