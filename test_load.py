import torch
import warnings
from model import create_model
from config import N_OUTPUT_DEPTHS, DEPTH_LEVELS_M

print("Loading model on CPU...")
device = torch.device("cpu")
model = create_model().to(device)
try:
    model.load_state_dict(torch.load("checkpoints/best_ocean_model_finetuned.pt", map_location=device), strict=False)
    print("Model loaded successfully!")
    
    # Run a dummy forward pass to make sure GELU works on CPU
    dummy = torch.randn(1, 7, 101, 241).to(device)
    with torch.no_grad():
        out = model(dummy)
    print("Dummy inference successful!")
except Exception as e:
    print(f"Error: {e}")
