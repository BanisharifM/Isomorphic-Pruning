import torch
import timm

# path = "results/pruned/ResNet101/final_pruned_resnet101_imagenet_rev12_ratio0.447_weights.pth"
# try:
#     obj = torch.load(path)
#     print("SUCCESS: Loaded.")
#     print("Type:", type(obj))
# except Exception as e:
#     print("FAILED:", e)




# Reconstruct the architecture (must match original)
model = timm.create_model("resnet101", pretrained=False, num_classes=1000)  # Change num_classes if needed

# Load weights
state_dict = torch.load("results/pruned/ResNet101/final_pruned_resnet101_imagenet_rev12_ratio0.447_full.pt", map_location='cpu')
model.load_state_dict(state_dict)
print("Model loaded successfully.")