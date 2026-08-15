import torch
import clip

# 1. Set up the device (Use GPU if available, otherwise fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the pre-trained CLIP model and its preprocessing pipeline
# We are using ViT-B/32 as planned in your proposal
print("Downloading and loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

print("CLIP Model loaded successfully!")
