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
RAW_SOURCES = [
    {
        "name": "trash_dataset",
        "path": "data/raw_downloads/underwater-trash", 
        "limit": 1000
    },
    {
        "name": "duo_dataset",
        "path": "data/raw_downloads/duo", 
        "limit": 1000
    },
    {
        "name": "sub_dataset",
        "path": "data/raw_downloads/underwater-submarines", 
        "limit": 500
    },
    {"name": "sub_dataset_2", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_3", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_4", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_5", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_6", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_7", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_8", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_9", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {"name": "sub_dataset_10", "path": "data/raw_downloads/underwater-submarines", "limit": 100},
    {
    "name": "real_diver_dataset",
    "path": "data/raw_downloads/diver-real", 
    "limit": 600
    }
]

# 2. Define the Intelligent Mapping Rules
# Format: "original_class_name": target_class_id
# Target IDs: 0=Mine, 1=Diver, 2=Submarine, -1=Ignore(Background)
# Note: These keywords match standard dataset class names. 

SMART_MAPPING = {
    # --- CLASS 0: MINES (Proxies: Trash/Debris) ---
    "trash": 0, "plastic": 0, "metal": 0, "debris": 0, 
    "bottle": 0, "can": 0, "cup": 0, "net": 0, "tire": 0,
    "bucket": 0, "drum": 0, "pipe": 0, "rubbish": 0,
    "container": 0, "fish trap": 0, "cage": 0,
    
    # --- CLASS 1: DIVERS (Humans) ---
    "diver": 1, "human": 1, "person": 1, "scuba": 1, 
    "scuba diver": 1, "swimmer": 1,

    # --- CLASS 2: SUBMARINES (ROVs/AUVs) ---
    "submarine": 2, "sub": 2, "rov": 2, "auv": 2, 
    "robot": 2, "uuv": 2, "vehicle": 2,

    # --- IGNORE (-1): Bio-life & Background ---
    # (These become empty background images for YOLO to learn 'nothingness')
    "fish": -1, "shark": -1, "turtle": -1, "plant": -1, 
    "coral": -1, "starfish": -1, "jellyfish": -1, "sea urchin": -1,
    "background": -1, "water": -1, "reef": -1, 
    # Specifics found in DUO dataset logs:
    "echinus": -1, "holothurian": -1, "scallop": -1, "shell": -1, "sea_star": -1
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

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
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
  0: mine
  1: diver
  2: submarine
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Smart Dataset Ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()