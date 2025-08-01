import torch
import timm


import torch, timm
from timm.models.convnext import ConvNeXt

# Allow-list the ConvNeXt class for unpickling
torch.serialization.add_safe_globals([ConvNeXt])

model = torch.load(
    "results/pruned/DeiT/Tiny/fine_tuned_deit_tiny_patch16_224_imagenet_rev1_ratio0.543_acc48.7.pth",
    map_location="cpu",
    weights_only=False      # ← turn OFF the new default
)
print("Loaded full pruned model:", type(model))



# try:
#     _ = torch.load("results/pruned/ConvNext/Small/final_pruned_convnext_small_imagenet_rev16_ratio0.536_full.pt", map_location='cpu')
#     print("[✅] Verified full model loads correctly.")
# except Exception as e:
#     print(f"[❌] Full model failed to load after saving: {e}")

# try:
#     _ = torch.load("results/pruned/ConvNext/Small/final_pruned_convnext_small_imagenet_rev16_ratio0.536_weights.pth", map_location='cpu')
#     print("[✅] Verified full model loads correctly.")
# except Exception as e:
#     print(f"[❌] Full model failed to load after saving: {e}")



# try:
#     test_model = timm.create_model("convnext_small", pretrained=False, num_classes=1000)
#     test_model.load_state_dict(torch.load("results/pruned/ConvNext/Small/final_pruned_convnext_small_imagenet_rev16_ratio0.536_full.pt", map_location='cpu'))
#     print("[✅] Verified weights load correctly into model.")
# except Exception as e:
#     print(f"[❌] Weights failed to load into model: {e}")

# try:
#     test_model = timm.create_model("convnext_small", pretrained=False, num_classes=1000)
#     test_model.load_state_dict(torch.load("results/pruned/ConvNext/Small/final_pruned_convnext_small_imagenet_rev16_ratio0.536_weights.pth", map_location='cpu'))
#     print("[✅] Verified weights load correctly into model.")
# except Exception as e:
#     print(f"[❌] Weights failed to load into model: {e}")


# path = "results/pruned/ResNet101/final_pruned_resnet101_imagenet_rev12_ratio0.447_weights.pth"
# try:
#     obj = torch.load(path)
#     print("SUCCESS: Loaded.")
#     print("Type:", type(obj))
# except Exception as e:
#     print("FAILED:", e)




# Reconstruct the architecture (must match original)
# model = timm.create_model("resnet101", pretrained=False, num_classes=1000)  # Change num_classes if needed

# Load weights
# state_dict = torch.load("results/pruned/ResNet101/final_pruned_resnet101_imagenet_rev12_ratio0.447_full.pt", map_location='cpu')
# model.load_state_dict(state_dict)
# print("Model loaded successfully.")