import sys
import os
import torch
from pathlib import Path

# Add core
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from core.models import FunieGANGenerator
from core.config import FUNIE_GAN_WEIGHTS

print(f"Root: {root_dir}")
print(f"Weights Path: {FUNIE_GAN_WEIGHTS}")
print(f"Exists: {os.path.exists(FUNIE_GAN_WEIGHTS)}")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = FunieGANGenerator().to(device)
    print("Model initialized")
    
    state_dict = torch.load(FUNIE_GAN_WEIGHTS, map_location=device)
    print("State dict loaded")
    
    model.load_state_dict(state_dict)
    print("M: GAN Weights loaded successfully!")

    print("Checking YOLO...")
    from ultralytics import YOLO
    from core.config import YOLO_WEIGHTS
    print(f"YOLO Path: {YOLO_WEIGHTS}")
    
    yolo = YOLO(YOLO_WEIGHTS)
    print("M: YOLO Loaded successfully")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
