import os
import cv2
import torch
import numpy as np
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# --- CONFIGURATION ---

# 1. Define your Source Folders
# --- CONFIGURATION FOR 4-CLASS DETECTION ---
# Target Classes: 
# 0 = Sea Mine 
# 1 = Diver 
# 2 = Drone (AUV/ROV) 
# 3 = Submarine

RAW_SOURCES = [
    # --- TIER 1: THREATS (CRITICAL: Taking 100% of available data) ---
    {
        "name": "mines_dataset",
        "path": "data/raw_dataset/mines",
        "limit": 221  # You have 221 images. Take them all.
    },
    {
        "name": "sub_dataset",
        "path": "data/raw_dataset/underwater-submarines", 
        "limit": 60   # You have 60 images. Take them all.
    },
    {
        "name": "drones_dataset",
        "path": "data/raw_dataset/underwater-drones", 
        "limit": 29   # You have 29 images. Take them all.
    },
    {
        "name": "real_diver_dataset",
        "path": "data/raw_dataset/diver-real", 
        "limit": 500  # You have ~700. 500 is enough to balance.
    },

    # --- TIER 2: ANOMALIES (SHAPES) ---
    # We cap these at ~500 so they don't overpower the Mines.
    {
        "name": "pipes_dataset",
        "path": "data/raw_dataset/underwater_pipes", 
        "limit": 500  # You have 7,900. Only need 500 for "Cylinder" concept.
    },
    {
        "name": "buoy_dataset",
        "path": "data/raw_dataset/buoys", 
        "limit": 489  # You have 489. Take them all for "Sphere".
    },
    {
        "name": "bins_dataset",
        "path": "data/raw_dataset/bins", 
        "limit": 500  # You have 1,192. 500 is perfect for "Box".
    },
    {
        "name": "duo_dataset",
        "path": "data/raw_dataset/duo", 
        "limit": 300  # Extra "Cylinders" (bottles/cans).
    },

    # --- TIER 3: BACKGROUND (NEGATIVE SAMPLES) ---
    # We need a 1:1 ratio of Background to Objects. 
    # Total Objects above = ~2,300. So we need ~2,000 background images.
    {
        "name": "brackish_bio",
        "path": "data/raw_dataset/brackish", 
        "limit": 1500 # You have 12,000. 1500 teaches "Ignore Fish".
    },
    {
        "name": "trash_dataset",
        "path": "data/raw_dataset/underwater-trash", 
        "limit": 500  # Extra negative samples (messy debris).
    }
]

# 2. Define the Intelligent Mapping Rules
# Format: "original_class_name": target_class_id
# NEW Target IDs: 0=Mine, 1=Diver, 2=Drone, 3=Submarine, -1=Ignore/Background

# --- 7-CLASS MAPPING STRATEGY ---
# 0: Mine        (Threat - Red)
# 1: Submarine   (Threat - Red)
# 2: Diver       (Threat - Red)
# 3: Drone       (Threat - Red)
# 4: Cylinder    (Anomaly - Yellow) -> Pipes, Round Traps, Tires
# 5: Sphere      (Anomaly - Yellow) -> Buoys
# 6: Box         (Anomaly - Yellow) -> Rectangular Traps
# -1: Background (Ignore - Gray)    -> Fish, Nets, Bio-life

SMART_MAPPING = {
    # --- TIER 1: THREATS (Red) ---
    "Mines": 0,                     # From 'mines'
    
    "submarine": 1, "unknowm": 1,   # From 'underwater-submarines' (fixed typo 'unknowm')
    
    "divers": 2,                    # From 'diver-real'
    
    "underwater_drones": 3,         # From 'drones'
    "unknown": 3,                   # 'unknown' in drone dataset is likely a drone
    
    # --- TIER 2: ANOMALIES (Yellow) ---
    
    # CLASS 4: CYLINDER (Pipes, Round Traps, Tires)
    "pipe": 4,                      # From 'underwater_pipes'
    "circular fish trap": 4,        # From 'trash' -> Perfect Cylinder Proxy
    "eel fish trap": 4,             # From 'trash' -> Long Tube Proxy
    "spring fish trap": 4,          # From 'trash' -> Coil/Tube Proxy
    "tire": 4,                      # From 'trash' -> Torus/Cylinder Proxy
    
    # CLASS 5: SPHERE (Buoys)
    "green_buoy": 5,                # From 'buoy'
    "orange_buoy": 5,               # From 'buoy'
    
    # CLASS 6: BOX (Rectangular Objects)
    "rectangular fish trap": 6,     # From 'trash' -> Perfect Box Proxy
    
    # --- TIER 3: BACKGROUND / NEGATIVE (Gray) ---
    # Map all bio-life and messy trash to -1 so the model ignores them.
    
    # From 'brackish' (Bio)
    "crab": -1, "fish": -1, "jellyfish": -1, 
    "shrimp": -1, "small_fish": -1, "starfish": -1,
    
    # From 'bins' (Which turned out to be bio!)
    "sawfish": -1, "shark": -1, 
    
    # From 'duo' (Bio)
    "echinus": -1, "holothurian": -1, "scallop": -1, 
    
    # From 'trash' (Unstructured Junk)
    "fish net": -1                  # Nets change shape, so treat as background
}
OUTPUT_DIR = "data/yolo_final_training"
GAN_WEIGHTS = "ml-engine/weights/funie_generator.pth"

# --- SETUP CODE ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml-engine")))
try:
    from img_enhancement.funie_gan.nets.funiegan import GeneratorFunieGAN
except ImportError:
    from ml_engine.core.models import GeneratorFunieGAN

def get_gan_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Loading FUnIE-GAN on {device}...")
    model = GeneratorFunieGAN().to(device)
    model.load_state_dict(torch.load(GAN_WEIGHTS, map_location=device))
    model.eval()
    return model, device

def load_source_classes(source_path):
    """Reads the data.yaml of the source to know which ID is which Name."""
    yaml_path = Path(source_path) / "data.yaml"
    if not yaml_path.exists():
        print(f"❌ Error: data.yaml not found in {source_path}")
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', {})
        # Handle list format vs dict format
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        return names

def process_and_save(img_path, label_path, output_root, model, device, source_id_to_name):
    """
    Enhances image AND intelligently maps labels.
    """
    # 1. Image Processing (GAN)
    img = cv2.imread(str(img_path))
    if img is None: return False

    img_256 = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = (img_tensor - 127.5) / 127.5
    
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)
    
    enhanced_tensor = (enhanced_tensor + 1.0) / 2.0
    enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
    final_img = cv2.resize(enhanced_bgr, (640, 640), interpolation=cv2.INTER_CUBIC)

    save_name = f"{img_path.parent.parent.name}_{img_path.name}"
    cv2.imwrite(os.path.join(output_root, "images", save_name), final_img)

    # 2. Smart Label Mapping
    if label_path.exists():
        new_lines = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    original_id = int(parts[0])
                    # Get the word (e.g., "fish" or "bottle")
                    original_name = source_id_to_name.get(original_id, "unknown").lower()
                    
                    # Check our rules
                    target_id = -1
                    # Partial match logic (e.g., "plastic_bag" matches "plastic")
                    for key, val in SMART_MAPPING.items():
                        if key in original_name:
                            target_id = val
                            break
                    
                    # If valid target (not -1/Ignore), save it
                    if target_id != -1:
                        parts[0] = str(target_id)
                        new_lines.append(" ".join(parts))
        
        # Only create label file if there are valid detections left
        if new_lines:
            label_save_name = save_name.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(output_root, "labels", label_save_name), 'w') as f:
                f.write("\n".join(new_lines))
    
    return True

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

def main():
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR, onerror=remove_readonly)
        except Exception as e:
            print(f"⚠️ Could not fully delete output dir: {e}. Continuing...")
            
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

    model, device = get_gan_model()

    for source in RAW_SOURCES:
        print(f"\n📂 Source: {source['name']}")
        
        # 1. Learn the Classes
        id_map = load_source_classes(source['path'])
        if not id_map: continue
        print(f"   ℹ️  Detected Classes: {list(id_map.values())}")

        # 2. Process Images
        path = Path(source['path']) / "train" / "images"
        all_images = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
        
        if not all_images:
            print(f"⚠️  No images found in {path}")
            continue

        selected = random.sample(all_images, min(len(all_images), source['limit']))
        print(f"   🚀 Processing {len(selected)} images...")
        
        for img_path in tqdm(selected):
            label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
            process_and_save(img_path, label_path, OUTPUT_DIR, model, device, id_map)

    # Generate Config
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images
val: images
names:
  0: Mine
  1: Submarine
  2: Diver
  3: Drone
  4: Cylinder
  5: Sphere
  6: Box
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Smart Dataset Ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()