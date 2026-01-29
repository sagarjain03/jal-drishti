import sys
import os
import cv2
import json
import base64
import unittest
import numpy as np

# Set up paths to import backend and ml-engine
# We are currently in jal-drishti/integration_tests/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # jal-drishti/

sys.path.insert(0, os.path.join(root_dir, "backend"))
sys.path.insert(0, os.path.join(root_dir, "ml-engine"))

from app.services.ml_service import ml_service
from app.schemas.response import AIResponse

# Configuration
TEST_IMAGES_DIR = os.path.join(root_dir, "ml-engine", "obj_detection", "test_images")
OUTPUT_DIR = os.path.join(root_dir, "ml-engine", "outputs", "integration_results")

class TestBackendMLIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Integration] Setting up Backend-ML Integration Test...")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Verify input directory
        if not os.path.exists(TEST_IMAGES_DIR):
            print(f"[Setup] WARNING: Test images dir not found at {TEST_IMAGES_DIR}")
            os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
            # Create dummy if empty
            dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(TEST_IMAGES_DIR, "dummy_integration.jpg"), dummy)
            print("[Setup] Created dummy test image.")
    
    def test_multi_image_flow_with_visualization(self):
        """
        Run the FULL Backend Pipeline on all images in the test_images folder.
        Saves the resulting images (with drawn boxes) to ml-engine/outputs/integration_results/
        """
        print(f"\n[MultiTest] Processing images from: {TEST_IMAGES_DIR}")
        
        # Get all valid images
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(valid_exts)]
        self.assertGreater(len(images), 0, "No test images found in the folder")
        
        print(f"Found {len(images)} images. Starting batch processing...\n")

        for img_name in images:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            
            # 1. Read Raw Image Bytes (Simulating WebSocket Payload)
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            # 2. Call Backend Service (The Real Pipeline)
            # This runs: Decode -> GAN -> YOLO -> Encode Base64 -> JSON Response
            resp = ml_service.process_frame(image_bytes)

            # 3. Validate Response Status
            self.assertEqual(resp["status"], "success")
            
            # 4. Decode the Result Image (Simulating Frontend Rendering)
            b64_str = resp["image_data"]
            img_data = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            final_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            self.assertIsNotNone(final_img, f"Failed to decode result for {img_name}")

            # 5. Draw Bounding Boxes (Simulating Frontend Overlay)
            # The backend returns 'detections', usually the frontend draws them.
            # We draw them here to verify visual accuracy.
            detections = resp.get("detections", [])
            for det in detections:
                bbox = det.get("bbox") # [x1, y1, x2, y2]
                label = det.get("label", "unknown")
                conf = det.get("confidence", 0.0)
                
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Draw Red Box
                    cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Draw Label Background
                    cv2.rectangle(final_img, (x1, y1-20), (x2, y1), (0, 0, 255), -1)
                    # Draw Text
                    text = f"{label} {conf:.2f}"
                    cv2.putText(final_img, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 6. Save the Final Result
            out_name = f"integrated_{img_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, final_img)
            
            print(f" > {img_name}: State={resp.get('state')} | Detections={len(detections)} | Saved to {out_name}")

if __name__ == "__main__":
    unittest.main()