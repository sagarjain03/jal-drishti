import torch
import numpy as np
import cv2
import datetime
import sys
import os

# Ensure we can import from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from img_enhancement.funie_gan.nets.funiegan import GeneratorFunieGAN as FunieGANGenerator
from .config import (
    FUNIE_GAN_WEIGHTS, YOLO_WEIGHTS, 
    CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    STATE_CONFIRMED_THREAT, STATE_POTENTIAL_ANOMALY, STATE_SAFE_MODE
)

print("[Core] Initializing JalDrishti Engine...")

class JalDrishtiEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Core] Using device: {self.device}")

        # 1. Load FUnIE-GAN
        self.gan = FunieGANGenerator().to(self.device)
        try:
            # Load weights if available, else warn (users need to place file)
            state_dict = torch.load(FUNIE_GAN_WEIGHTS, map_location=self.device)
            self.gan.load_state_dict(state_dict)
            print("[Core] FUnIE-GAN weights loaded.")
        except FileNotFoundError:
            print(f"[Core] WARNING: FUnIE-GAN weights not found at {FUNIE_GAN_WEIGHTS}. Using random weights.")
        except Exception as e:
            print(f"[Core] Error loading GAN weights: {e}")
        
        self.gan.eval()

        # 2. Load YOLOv8
        try:
            self.yolo = YOLO(YOLO_WEIGHTS)
            print("[Core] YOLOv8 model loaded.")
        except Exception as e:
            print(f"[Core] WARNING: Could not load YOLO weights at {YOLO_WEIGHTS}. Downloading/Using default yolov8n.pt")
            self.yolo = YOLO("yolov8n.pt") # Fallback to auto-download

        # Warmup
        print("[Core] Engine ready.")

    def validate_frame(self, frame):
        """Step 1: Frame Validity Gate"""
        if frame is None:
            return False, "Frame is None"
        if frame.size == 0:
            return False, "Empty frame"
        if len(frame.shape) != 3:
            return False, "Invalid dimensions"
        if frame.shape[2] != 3:
            return False, "Invalid channel count (Not RGB)"
        return True, None

    def infer(self, frame: np.ndarray):
        """
        Executes the 7-Step Phase-2 Pipeline.
        Input: RGB numpy array (H, W, 3)
        Output: Strict JSON Schema + Enhanced Frame (np.ndarray)
        """
        # --- Step 1: Gate ---
        valid, msg = self.validate_frame(frame)
        if not valid:
            print(f"[Core] Invalid frame: {msg}")
            return self._build_safe_response(), frame

        try:
            # --- Step 2: Pre-process (GAN Side) ---
            # Resize to 256x256
            original_h, original_w = frame.shape[:2]
            img_resized = cv2.resize(frame, (256, 256))
            
            # Normalize to [-1, 1]: (x - 127.5) / 127.5
            # Convert BGR (OpenCV) -> RGB (GAN expectation)
            img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_resized_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = (img_tensor - 127.5) / 127.5

            # --- Step 3: GAN Inference ---
            with torch.no_grad():
                enhanced_tensor = self.gan(img_tensor)

            # --- Step 4: Normalization Bridge (CRITICAL) ---
            # Convert [-1, 1] -> [0, 1]: (x + 1) / 2
            enhanced_tensor = (enhanced_tensor + 1.0) / 2.0
            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0) # Safety Clamp

            # Resize to YOLO input size (640x640)
            # YOLOv8 handles resizing internally, but we pass the enhanced tensor/image
            # Convert back to numpy uint8 for YOLO / Visuals
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np_uint8 = (enhanced_np * 255).astype(np.uint8)
            
            # Convert RGB (GAN) -> BGR (OpenCV/YOLO expectation)
            enhanced_np_uint8 = cv2.cvtColor(enhanced_np_uint8, cv2.COLOR_RGB2BGR)
            
            # Optional: Upscale for clearer detection if needed, or pass 256x256
            # Plan says: Resize -> 640x640 (bilinear)
            enhanced_cv = cv2.resize(enhanced_np_uint8, (640, 640))

            # --- Step 5: YOLOv8 Inference ---
            results = self.yolo.predict(enhanced_cv, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            # --- Step 6: Confidence & Safety Logic ---
            detections = []
            max_conf = 0.0
            
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo.names[cls]
                
                # We can enforce "anomaly" label per plan constraint, or keep real labels
                # Plan says: Label (generic: anomaly)
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 3),
                    "label": "anomaly"  # Enforcing generic label
                })
                
                if conf > max_conf:
                    max_conf = conf

            # Determine System State
            if max_conf > HIGH_CONFIDENCE_THRESHOLD:
                state = STATE_CONFIRMED_THREAT
            elif max_conf > CONFIDENCE_THRESHOLD:
                state = STATE_POTENTIAL_ANOMALY
            else:
                state = STATE_SAFE_MODE # Even with no detections or low confidence
                
            # --- Step 7: Output Contract ---
            response = {
                "timestamp": datetime.datetime.now().isoformat(),
                "state": state,
                "max_confidence": round(max_conf, 3),
                "detections": detections
            }
            
            return response, enhanced_cv

        except Exception as e:
            print(f"[Core] Pipeline Error: {e}")
            return self._build_safe_response(), frame

    def _build_safe_response(self):
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "state": STATE_SAFE_MODE,
            "max_confidence": 0.0,
            "detections": []
        }
