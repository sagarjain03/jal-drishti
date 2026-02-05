import torch
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import argparse
import sys

# --- PATH SETUP (To find your model definition) ---
# current_dir = ml-engine/img_enhancement
current_dir = Path(__file__).resolve().parent
# project_root = jal-drishti (assuming ml-engine is at root of project)
# If current_dir is img_enhancement, parent is ml-engine. parent.parent is jal-drishti.
# Let's be robust.
if current_dir.name == "img_enhancement":
    PROJECT_ROOT = current_dir.parent.parent
elif current_dir.name == "ml-engine":
    PROJECT_ROOT = current_dir.parent
else:
    # Fallback/Guess
    PROJECT_ROOT = current_dir.parent.parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "ml-engine")) # Ensure ml-engine is importable

try:
    from ml_engine.core.models import GeneratorFunieGAN
except ImportError:
    # Fallback if structure is slightly different
    sys.path.append(str(PROJECT_ROOT / "ml-engine"))
    from img_enhancement.funie_gan.nets.funiegan import GeneratorFunieGAN

# ---------- DEFAULT CONFIG ----------
# ---------- DEFAULT CONFIG ----------
# Use absolute paths based on PROJECT_ROOT
DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "data" / "raw_dataset" / "underwater-trash" / "train" / "images")
DEFAULT_WEIGHTS_PATH = str(PROJECT_ROOT / "ml-engine" / "weights" / "funie_generator.pth")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "evaluation_results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expose these as globals for external scripts (like evaluate_pipeline.py)
INPUT_DIR = DEFAULT_INPUT_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH
# ------------------------------------

# --- HELPER: CLAHE FUNCTION ---
def apply_clahe(img_bgr):
    """
    Applies Contrast Limited Adaptive Histogram Equalization 
    to the Luminance channel.
    """
    # 1. Convert BGR to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Apply CLAHE to L-channel (Lightness)
    # clipLimit=2.0 is the standard "safe" value. Higher = more contrast but more noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # 3. Merge and convert back to BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    final_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return final_img

def main():
    parser = argparse.ArgumentParser(description="Run FUnIE-GAN + CLAHE Hybrid Pipeline")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR, help="Path to input images")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to save enhanced images")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to GAN weights")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    
    args = parser.parse_args()
    
    global INPUT_DIR, OUTPUT_DIR, WEIGHTS_PATH
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    WEIGHTS_PATH = args.weights
    LIMIT = args.limit

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in Path(INPUT_DIR).iterdir() if f.suffix.lower() in image_extensions]
    total_images = len(image_files)

    print(f"🔄 Found {total_images} images to process...")
    print(f"⚡ Device: {DEVICE}")
    print(f"🚀 Pipeline: FUnIE-GAN + CLAHE (Hybrid)")
    print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load model
    model = GeneratorFunieGAN().to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Weights not found at {WEIGHTS_PATH}")
        return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    start_time = datetime.now()
    processed = 0
    failed = 0

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        try:
            # 1. Load image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️  [{idx}/{total_images}] Failed to read: {image_path.name}")
                failed += 1
                continue
            
            # 2. Preprocess for GAN (Resize to 256x256)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256))
            
            # Normalize to [-1,1]
            img_tensor = (img_resized.astype(np.float32) - 127.5) / 127.5
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # 3. GAN Inference
            with torch.no_grad():
                out_tensor = model(img_tensor)
            
            # 4. De-normalize & Convert to Numpy
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_uint8 = ((out_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # 5. Resize to Target Size (640x640 for YOLO or Display)
            # Cubic interpolation is better for upscaling details
            out_final = cv2.resize(out_uint8, (640, 640), interpolation=cv2.INTER_CUBIC)
            
            # 6. Convert to BGR (OpenCV Format)
            out_bgr = cv2.cvtColor(out_final, cv2.COLOR_RGB2BGR)

            # 7. APPLY CLAHE (The "Engineer" Step)
            out_hybrid = apply_clahe(out_bgr)
            
            # 8. Save output
            output_path = Path(OUTPUT_DIR) / image_path.name
            cv2.imwrite(str(output_path), out_hybrid)
            
            cv2.imwrite(str(output_path), out_hybrid)
            
            processed += 1
            if idx % 50 == 0 or idx == total_images:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / idx
                eta_remaining = avg_time * (total_images - idx)
                print(f"✅ [{idx}/{total_images}] Processed: {image_path.name} | Avg: {avg_time:.2f}s/img | ETA: {eta_remaining:.1f}s")

            if LIMIT and processed >= LIMIT:
                print(f"🛑 Limit of {LIMIT} images reached.")
                break
        
        except Exception as e:
            print(f"❌ [{idx}/{total_images}] Error processing {image_path.name}: {str(e)}")
            failed += 1

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"📊 PROCESSING COMPLETE (HYBRID PIPELINE)")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed} images")
    print(f"⚠️  Failed: {failed} images")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Hybrid Output saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()