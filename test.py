import torch
from timm.models.vision_transformer import VisionTransformer
from timm.layers.patch_embed import PatchEmbed

# Trust these timm classes (safe if file is from your source)
torch.serialization.add_safe_globals([
    VisionTransformer,
    PatchEmbed
])

# Now load the checkpoint
ckpt = torch.load("/work/hdd/bewo/mahdi/agentic_prune/models/Final/final_pruned_deit_small_patch16_224_imagenet_rev1_ratio0.077.pt")
print("✅ Checkpoint loaded successfully!")
print("Type:", type(ckpt))
