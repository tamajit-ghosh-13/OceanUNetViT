import torch
import threading
from model import create_model
from config import N_INPUT_CHANNELS

device = torch.device('mps')
model = create_model(in_channels=N_INPUT_CHANNELS).to(device)
dummy = torch.randn(1, N_INPUT_CHANNELS, 101, 241, device=device)

def run_inference():
    try:
        with torch.no_grad():
            out = model(dummy)
        print("Success on background thread!")
    except Exception as e:
        print("Exception:", e)

t = threading.Thread(target=run_inference)
t.start()
t.join()
