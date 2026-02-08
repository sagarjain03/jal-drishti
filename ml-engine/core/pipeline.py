import torch
import torch.nn.functional as F
import numpy as np
import cv2
import datetime
import sys
import os
import logging

# Ensure we can import from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from img_enhancement.funie_gan.nets.funiegan import GeneratorFunieGAN as FunieGANGenerator
from .config import (
    FUNIE_GAN_WEIGHTS, YOLO_WEIGHTS, 
    CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    STATE_CONFIRMED_THREAT, STATE_POTENTIAL_ANOMALY, STATE_SAFE_MODE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("[Core] Initializing JalDrishti Engine...")

class JalDrishtiEngine:
    def __init__(self, use_gpu=True, use_fp16=True):
        """
        Initialize the ML Pipeline with GPU/FP16 support.
        
        Args:
            use_gpu (bool): Enable GPU if available (with CPU fallback)
            use_fp16 (bool): Enable FP16 half-precision inference
        """
        self.use_fp16 = use_fp16
        self.device = self._init_device(use_gpu)
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        
        logger.info(f"[Core] Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"[Core] GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"[Core] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"[Core] FP16 Enabled: {self.use_fp16}")
        
        # 1. Load FUnIE-GAN
        self.gan = FunieGANGenerator().to(self.device)
        try:
            # Load weights if available, else warn (users need to place file)
            state_dict = torch.load(FUNIE_GAN_WEIGHTS, map_location=self.device)
            self.gan.load_state_dict(state_dict)
            logger.info("[Core] FUnIE-GAN weights loaded.")
        except FileNotFoundError:
            logger.warning(f"[Core] FUnIE-GAN weights not found at {FUNIE_GAN_WEIGHTS}. Using random weights.")
        except Exception as e:
            logger.error(f"[Core] Error loading GAN weights: {e}")
        
        self.gan.eval()

        # 2. Load YOLOv8
        try:
            self.yolo = YOLO(YOLO_WEIGHTS)
            self.yolo.to(self.device)
            logger.info(f"[Core] YOLOv8 model loaded. Classes: {self.yolo.names}")
        except Exception as e:
            logger.warning(f"[Core] Could not load YOLO weights at {YOLO_WEIGHTS}. Downloading/Using default yolov8n.pt")
            self.yolo = YOLO("yolov8n.pt") # Fallback to auto-download
            self.yolo.to(self.device)

        # 2a. Track History for Persistence
        self.track_history = {} # track_id -> frames_seen

        # Warmup
        logger.info("[Core] Engine ready.")
    
    def _init_device(self, use_gpu=True):
        """
        Initialize and detect device with fallback to CPU.
        
        Returns:
            torch.device: Selected device (cuda or cpu)
        """
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("[Core] GPU (CUDA) detected and enabled")
            return device
        elif use_gpu:
            logger.warning("[Core] GPU requested but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
        else:
            logger.info("[Core] CPU mode selected")
            return torch.device("cpu")

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

    def _build_safe_response(self):
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "state": STATE_SAFE_MODE,
            "max_confidence": 0.0,
            "detections": [],
            "latency_ms": 0.0
        }

    def infer(self, frame: np.ndarray):
        """
        Executes the Active Analyst Pipeline (Single-Stream Tracking).
        
        Phase 1: Concept - "Active Analyst"
        Phase 2: Setup - Low Conf Inference (0.10)
        Phase 3: Logic - 3-Bucket Sorting
        Phase 4: Visuals - "Ghost Box" (Hidden for now per user request)
        Phase 5: Execution - Strict Rules
        
        Input: RGB numpy array (H, W, 3)
        Output: Strict JSON Schema + Annotated Frame (np.ndarray)
        """
        import time
        start_time = time.time()
        
        # --- Step 1: Gate ---
        valid, msg = self.validate_frame(frame)
        if not valid:
            logger.warning(f"[Core] Invalid frame: {msg}")
            return self._build_safe_response(), frame

        try:
            # --- Step 2: Pre-process (GAN Side) ---
            # We still run GAN for the visuals, even if we infer on Raw
            # Resize to 256x256 (Native GAN resolution)
            original_h, original_w = frame.shape[:2]
            img_resized = cv2.resize(frame, (256, 256))
            
            # Normalize to [-1, 1]
            img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_resized_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = (img_tensor - 127.5) / 127.5

            # --- Step 3: GAN Inference ---
            with torch.no_grad():
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        enhanced_tensor = self.gan(img_tensor)
                else:
                    enhanced_tensor = self.gan(img_tensor)

            # --- Step 4: Post-Process (GAN) ---
            enhanced_tensor = (enhanced_tensor + 1.0) / 2.0
            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
            
            # Resize back to original
            enhanced_tensor = F.interpolate(
                enhanced_tensor.unsqueeze(0) if len(enhanced_tensor.shape) == 3 else enhanced_tensor, 
                size=(original_h, original_w), 
                mode='bilinear', 
                align_corners=False
            )
            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np_uint8 = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
            
            # Convert RGB (GAN) -> BGR (OpenCV)
            enhanced_bgr = cv2.cvtColor(enhanced_np_uint8, cv2.COLOR_RGB2BGR)

            # --- Step 5: CLAHE (Enhancement) ---
            enhanced_final = self.apply_clahe(enhanced_bgr)

            # --- Step 6: Single-Stream Inference (Active Analyst) ---
            # "We lower the bar to 10% to catch debris, rocks, and faint shadows."
            # We track on RAW frame for stability.
            results = self.yolo.track(
                frame, 
                persist=True, 
                conf=0.10, 
                verbose=False,
                device=self.device
            )
            
            annotated_frame = enhanced_final.copy()
            detections_payload = []
            max_conf = 0.0
            
            # buckets
            threats = []   # Tier 1 (Red)
            anomalies = [] # Tier 2 (Yellow)
            neutrals = []  # Tier 3 (Hidden)

            # Smart Mapping for Sorting
            SPECIFIC_THREATS = [0, 1, 2, 3] # Mine, Sub, Diver, Drone
            GEOMETRIC_SHAPES = [4, 5, 6]    # Pipe/Cylinder, Buoy/Sphere, Trap/Box
            
            # Reset track history if no detections (optional, or rely on max_age logic, 
            # here we just keep growing it for simplicity within session)
            current_ids = []

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, classes):
                    current_ids.append(track_id)
                    
                    # Update Persistence
                    self.track_history[track_id] = self.track_history.get(track_id, 0) + 1
                    frames_seen = self.track_history[track_id]
                    
                    if conf > max_conf:
                        max_conf = conf

                    # --- BUCKET 1: THREATS (High Conf Specifics) ---
                    # Red Box
                    if cls_id in SPECIFIC_THREATS and conf > 0.60:
                        threats.append((box, track_id, conf, cls_id))

                    # --- BUCKET 2: ANOMALIES (Shapes OR Low Conf Specifics) ---
                    # Yellow Box
                    elif (cls_id in GEOMETRIC_SHAPES and conf > 0.35) or \
                         (cls_id in SPECIFIC_THREATS and 0.35 < conf <= 0.60):
                        anomalies.append((box, track_id, conf, cls_id))
                        
                    # --- BUCKET 3: NEUTRALS (Everything Else) ---
                    # Hidden (filtered out visually) but logically present
                    else:
                        neutrals.append((box, track_id, conf, cls_id))

            # --- VISUALIZATION STRATEGY ---
            
            # Helper to draw box
            def draw_detection(box, label, color, thickness=2):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 1. Text Settings (Smaller font per your request)
                font_scale = 0.4
                font_thickness = 1
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # 2. Smart Position Logic (Fixes "Missing Label" bug)
                # If box is at the very top (y1 < 20), draw text INSIDE the box
                if y1 - 20 < 0:
                    text_y = y1 + h + 5
                    bg_y1 = y1
                    bg_y2 = y1 + h + 10
                else:
                    text_y = y1 - 5
                    bg_y1 = y1 - h - 10
                    bg_y2 = y1
                
                # Draw Background
                cv2.rectangle(annotated_frame, (x1, text_y - h - 5), (x1 + w, text_y + 5), color, -1)
                
                # Draw Text
                cv2.putText(annotated_frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)

            # 1. DRAW THREATS (RED)
            for (box, track_id, conf, cls_id) in threats:
                # "Fix: Emulate Strict Labeling" - No ghost boxes
                label_name = self.yolo.names[int(cls_id)].upper()
                label = f"{label_name} {int(conf*100)}%"
                draw_detection(box, label, (0, 0, 255), thickness=2)
                
                # Add to payload
                detections_payload.append({
                    "bbox": [int(x) for x in box],
                    "confidence": round(float(conf), 3),
                    "label": label_name,
                    "type": "THREAT"
                })

            # 2. DRAW ANOMALIES (YELLOW)
            for (box, track_id, conf, cls_id) in anomalies:
                label_name = self.yolo.names[int(cls_id)]
                label = f"ANOMALY: {label_name}"
                draw_detection(box, label, (0, 255, 255), thickness=2)
                
                detections_payload.append({
                    "bbox": [int(x) for x in box],
                    "confidence": round(float(conf), 3),
                    "label": label_name,
                    "type": "ANOMALY"
                })

            # 3. NEUTRALS (HIDDEN)
            # We do NOT draw them. We do NOT add them to the payload (unless requested for debug).
            # They stay as silent tracks to prevent them from flickering into anomalies.

            # Determine System State
            if len(threats) > 0:
                state = STATE_CONFIRMED_THREAT
            elif len(anomalies) > 0:
                state = STATE_POTENTIAL_ANOMALY
            else:
                state = STATE_SAFE_MODE
                
            # --- Output Contract ---
            latency_ms = (time.time() - start_time) * 1000
            response = {
                "timestamp": datetime.datetime.now().isoformat(),
                "state": state,
                "max_confidence": round(float(max_conf), 3),
                "detections": detections_payload,
                "latency_ms": round(latency_ms, 2)
            }
            
            return response, annotated_frame

        except Exception as e:
            logger.error(f"[Core] Pipeline Error: {e}", exc_info=True)
            return self._build_safe_response(), frame

    def apply_clahe(self, img_bgr):
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
