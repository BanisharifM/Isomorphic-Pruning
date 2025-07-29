import torch
import argparse
import timm
from torch.serialization import add_safe_globals
from timm.models.vision_transformer import VisionTransformer

def load_model(path):
    # Allow VisionTransformer classes for unpickling
    add_safe_globals([VisionTransformer])

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Case 1: full checkpoint with model_state_dict or model
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            # If checkpoint["model"] is a model object
            if isinstance(checkpoint["model"], torch.nn.Module):
                return checkpoint["model"].state_dict()
            # Or if it's already a state dict
            return checkpoint["model"]

    # Case 2: plain model object
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    # Case 3: already a state_dict
    if isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        return checkpoint


    raise ValueError(f"Cannot find state_dict in: {path}")

def compare_models(model1_path, model2_path):
    print(f"\nLoading model1: {model1_path}")
    state_dict1 = load_model(model1_path)

    print(f"Loading model2: {model2_path}")
    state_dict2 = load_model(model2_path)

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    print("\n=== Key Comparison ===")
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    in_both = keys1 & keys2

    print(f"✅ Shared parameters: {len(in_both)}")
    if only_in_1:
        print(f"❌ Parameters only in {model1_path}:")
        for k in sorted(only_in_1):
            print(f"   {k}")
    if only_in_2:
        print(f"❌ Parameters only in {model2_path}:")
        for k in sorted(only_in_2):
            print(f"   {k}")

    print("\n=== Shape Comparison ===")
    for k in in_both:
        shape1 = tuple(state_dict1[k].shape)
        shape2 = tuple(state_dict2[k].shape)
        if shape1 != shape2:
            print(f"🔺 Shape mismatch for {k}: {shape1} vs {shape2}")

    print("\nComparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True, help="Path to your pruned model")
    parser.add_argument("--model2", type=str, required=True, help="Path to the isomorphic-pruned model")
    args = parser.parse_args()

    compare_models(args.model1, args.model2)
