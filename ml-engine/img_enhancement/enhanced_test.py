import os
import cv2
import base64
import numpy as np
from app.services.ml_service import ml_service

# Paths configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "ml-engine", "img_enhancement", "test_images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ml-engine", "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of test images (you can add more files here)
image_files = ["raw_test.jpg", "test.jpeg"]

def process_and_save(image_name: str):
    img_path = os.path.join(TEST_IMAGES_DIR, image_name)
    if not os.path.exists(img_path):
        print(f"[Enhanced Test] WARNING: Image not found: {img_path}")
        return
    # Read image as bytes
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    # Run through the ML service (which includes GAN + YOLO)
    result = ml_service.process_frame(img_bytes)
    # Decode the enhanced image from base64
    enhanced_b64 = result.get("image_data")
    if not enhanced_b64:
        print(f"[Enhanced Test] No enhanced image returned for {image_name}")
        return
    img_data = base64.b64decode(enhanced_b64)
    nparr = np.frombuffer(img_data, np.uint8)
    enhanced_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if enhanced_img is None:
        print(f"[Enhanced Test] Failed to decode enhanced image for {image_name}")
        return
    # Save enhanced image
    out_path = os.path.join(OUTPUT_DIR, f"enhanced_{image_name}")
    cv2.imwrite(out_path, enhanced_img)
    print(f"[Enhanced Test] Saved enhanced image to {out_path}")
    # Print detections summary
    detections = result.get("detections", [])
    print(f"[Enhanced Test] Detections for {image_name} ({len(detections)}): {detections}\n")

if __name__ == "__main__":
    for img_file in image_files:
        process_and_save(img_file)
