import sys
import os
import cv2
import numpy as np
import json
import torch
import unittest
from pathlib import Path

# Add core to path (Assumes this file is in ml-engine/tests/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pipeline import JalDrishtiEngine
from core.config import STATE_SAFE_MODE, STATE_POTENTIAL_ANOMALY

# --- CONFIGURATION ---
# Path to the folder containing your test images
TEST_IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../obj_detection/test_images"))
# Path where results will be saved
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))

class TestJalDrishtiEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Setup] Initializing Engine...")
        cls.engine = JalDrishtiEngine()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Verify Test Directory Exists
        if not os.path.exists(TEST_IMAGES_DIR):
            print(f"[Setup] WARNING: Test directory not found at {TEST_IMAGES_DIR}")
            os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
            # Create one dummy image so tests don't fail completely
            dummy_path = os.path.join(TEST_IMAGES_DIR, "dummy_gen.jpg")
            cv2.imwrite(dummy_path, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            print(f"[Setup] Created dummy test image at {dummy_path}")

    def test_01_component_loading(self):
        """Test if models loaded correctly"""
        print("\n[Test 1] Component Loading")
        self.assertIsNotNone(self.engine.gan, "GAN model should be loaded")
        self.assertIsNotNone(self.engine.yolo, "YOLO model should be loaded")
        self.assertFalse(self.engine.gan.training, "GAN should be in eval mode")

    def test_02_gate_logic(self):
        """Test Input Validation Gate"""
        print("\n[Test 2] Gate Logic")
        valid, msg = self.engine.validate_frame(None)
        self.assertFalse(valid)
        valid, msg = self.engine.validate_frame(np.zeros((100, 100), dtype=np.uint8))
        self.assertFalse(valid)
        valid, msg = self.engine.validate_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertTrue(valid)

    def test_03_preprocessing(self):
        """Test Preprocessing Logic (Resize/Normalize)"""
        print("\n[Test 3] Preprocessing")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = cv2.resize(frame, (256, 256))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
        normalized = (tensor - 127.5) / 127.5
        self.assertEqual(normalized.shape, (1, 3, 256, 256))
        self.assertTrue(normalized.min() >= -1.0)

    def test_04_batch_inference(self):
        """
        Run Full Pipeline on ALL images in the test_images folder.
        """
        print(f"\n[Test 4] Batch Inference on folder: {TEST_IMAGES_DIR}")
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            self.fail("No images found in test_images folder!")

        print(f"Found {len(image_files)} images to process.")
        
        for img_name in image_files:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            print(f"   > Processing: {img_name}...")
            
            frame = cv2.imread(img_path)
            self.assertIsNotNone(frame, f"Failed to load {img_name}")

            # --- RUN INFERENCE ---
            response, enhanced_frame = self.engine.infer(frame)

            # --- VERIFY RESPONSE ---
            self.assertIn("timestamp", response)
            self.assertIn("state", response)
            self.assertIn("detections", response)
            
            # --- DRAW BOXES ON OUTPUT ---
            # Draw the detections on the enhanced frame so you can visually check them
            for det in response["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.2f}"
                # Draw Box
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw Label
                cv2.putText(enhanced_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- SAVE RESULT ---
            out_name = f"result_{img_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, enhanced_frame)
            print(f"     [Saved] -> {out_name} | State: {response['state']} | Detections: {len(response['detections'])}")

    def test_05_safety_logic(self):
        """Test Safety Logic with blank frame"""
        print("\n[Test 5] Safety Logic (Blank Frame)")
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resp, _ = self.engine.infer(black_frame)
        self.assertEqual(resp['state'], STATE_SAFE_MODE)

if __name__ == '__main__':
    unittest.main()