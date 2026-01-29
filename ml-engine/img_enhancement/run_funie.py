import torch
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from funie_gan.nets.funiegan import GeneratorFunieGAN

# ---------- CONFIG ----------
INPUT_DIR = "../data/enhancement/raw"
WEIGHTS_PATH = "../weights/funie_generator.pth"
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [f for f in Path(INPUT_DIR).iterdir() if f.suffix.lower() in image_extensions]
total_images = len(image_files)

print(f"🔄 Found {total_images} images to process...")
print(f"Device: {DEVICE}")
print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load model
model = GeneratorFunieGAN().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

start_time = datetime.now()
processed = 0
failed = 0

# Process each image
for idx, image_path in enumerate(image_files, 1):
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️  [{idx}/{total_images}] Failed to read: {image_path.name}")
            failed += 1
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # Normalize to [-1,1]
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            out = model(img)
        
        # De-normalize
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = ((out + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Save output
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        output_path = Path(OUTPUT_DIR) / image_path.name
        cv2.imwrite(str(output_path), out)
        
        processed += 1
        if idx % 100 == 0 or idx == total_images:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / idx
            eta_remaining = avg_time * (total_images - idx)
            print(f"✅ [{idx}/{total_images}] Processed: {image_path.name} | Avg: {avg_time:.2f}s/img | ETA: {eta_remaining:.1f}s")
    
    except Exception as e:
        print(f"❌ [{idx}/{total_images}] Error processing {image_path.name}: {str(e)}")
        failed += 1

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"📊 PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"✅ Successfully processed: {processed} images")
print(f"⚠️  Failed: {failed} images")
print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"⏱️  Average time per image: {total_time/total_images:.2f} seconds")
print(f"📁 Output saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
