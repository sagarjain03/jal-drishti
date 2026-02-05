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
        Executes the 7-Step Phase-2 Pipeline with GPU/FP16 optimization.
        Executes the Dual-Stream Pipeline.
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
            # Resize to 256x256 (Native GAN resolution)
            original_h, original_w = frame.shape[:2]
            img_resized = cv2.resize(frame, (256, 256))
            
            # Normalize to [-1, 1]
            img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_resized_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = (img_tensor - 127.5) / 127.5

            # --- Step 3: GAN Inference with FP16 Support ---
            with torch.no_grad():
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        enhanced_tensor = self.gan(img_tensor)
                else:
                    enhanced_tensor = self.gan(img_tensor)

            # --- Step 4: Post-Process ---
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

            # --- Step 6: Dual-Stream YOLO Inference (OPTIMIZED) ---
            
            # Combine both images into a list for Batch Processing
            # This runs 2x faster than calling predict() twice
            batch_images = [frame, enhanced_final]
            
            # LOGIC PRESERVATION: Determine FP16 (Half-Precision) setting
            # This ensures we keep the optimization from the code we just deleted
            do_half = (self.use_fp16 and self.device.type == "cuda")
            
            # Run Inference once
            # We use conf=0.15 (Low) intentionally so our custom 'process_results' 
            # function can handle the specific class thresholds (e.g. Drone=0.15)
            batch_results = self.yolo.predict(
                batch_images, 
                verbose=False, 
                conf=0.15, 
                device=self.device, 
                half=do_half  # <--- Optimization Preserved Here
            )
            
            # Split results back
            results_raw = [batch_results[0]]      # Result for 'frame'
            results_enhanced = [batch_results[1]] # Result for 'enhanced_final''
            
            # --- Step 7: Merge & Annotate ---
            detections = []
            max_conf = 0.0
            
            # We draw on the Enhanced Frame because it looks better (User requirement)
            # OR we draw on Raw? User said "Draw on Enhanced Feed because it looks cooler".
            annotated_frame = enhanced_final.copy()
            
            # Helper to process results with CLASS-SPECIFIC THRESHOLDS
            raw_detections_list = []

            def process_results(results, source_tag):
                nonlocal max_conf
                if not results: return
                
                result = results[0]
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # 🛡️ GATEKEEPER LOGIC 🛡️
                    # Har class ka apna cutoff set karo
                    min_thresh = 0.25 # Default
                    
                    if cls == 2:   # DRONE (Mushkil hai, allow low confidence)
                        min_thresh = 0.15
                    elif cls == 1: # DIVER (Insaan hai, model sure hona chahiye)
                        min_thresh = 0.55  # 👈 Isse False Positives kam honge
                    elif cls == 0: # MINE
                        min_thresh = 0.40
                    elif cls == 3: # SUBMARINE
                        min_thresh = 0.40
                        
                    # Agar confidence cutoff se kam hai, toh ignore karo
                    if conf < min_thresh:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = self.yolo.names[cls]
                    
                    # Store in intermediate list for Smart NMS
                    raw_detections_list.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": conf,
                        "cls": cls,
                        "label": label,
                        "source": source_tag
                    })
                    
                    if conf > max_conf:
                        max_conf = conf

            # Process both streams
            process_results(results_raw, "RAW_SENSOR")
            process_results(results_enhanced, "AI_ENHANCED")
            
            # --- Step 7.5: SMART NMS (The Fix) ---
            final_detections = self.clean_detections(raw_detections_list)

            # --- Step 7.6: Draw & Format ---
            for det in final_detections:
                x1, y1, x2, y2 = det["box"]
                label = det["label"]
                conf = det["conf"]
                source_tag = det["source"]
                
                # Add to final response list
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 3),
                    "label": label,
                    "source": source_tag
                })

                # --- ANNOTATION ---
                # Green for Raw, Orange for AI
                color = (0, 255, 0) if source_tag == "RAW_SENSOR" else (0, 140, 255)
                
                # Draw Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label (Simplified, NO source tag)
                label_text = f"{label} {int(conf*100)}%"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Ensure text never goes off-screen
                text_y = max(y1 - 5, 20)
                
                cv2.rectangle(annotated_frame, (x1, text_y - 15), (x1 + w, text_y + 5), color, -1)
                
                # Black text (0,0,0) for better contrast on bright Green/Orange
                cv2.putText(annotated_frame, label_text, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Determine System State
            if max_conf > HIGH_CONFIDENCE_THRESHOLD:
                state = STATE_CONFIRMED_THREAT
            elif max_conf > CONFIDENCE_THRESHOLD:
                state = STATE_POTENTIAL_ANOMALY
            else:
                state = STATE_SAFE_MODE
                
            # --- Step 7: Output Contract ---
            latency_ms = (time.time() - start_time) * 1000
            response = {
                "timestamp": datetime.datetime.now().isoformat(),
                "state": state,
                "max_confidence": round(max_conf, 3),
                "detections": detections,
                "latency_ms": round(latency_ms, 2)
            }
            
            # Return the annotated enhanced frame directly (No blue filter!)
            return response, annotated_frame

        except Exception as e:
            logger.error(f"[Core] Pipeline Error: {e}", exc_info=True)
            return self._build_safe_response(), frame


    def clean_detections(self, detections, iou_threshold=0.4):
        """
        Smart Cleaner with 'Diver Priority'.
        Rule: If Diver (1) and Submarine (3) overlap, KILL the Submarine.
        """
        # 1. Sort by confidence (Highest first)
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        remove_indices = set()
        
        for i in range(len(detections)):
            if i in remove_indices: continue
            
            for j in range(i + 1, len(detections)):
                if j in remove_indices: continue
                
                # --- IoU Calculation ---
                boxA = detections[i]['box']
                boxB = detections[j]['box']
                
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)
                
                # --- LOGIC: CONFLICT RESOLUTION ---
                if iou > iou_threshold:
                    cls1 = detections[i]['cls']
                    cls2 = detections[j]['cls']
                    
                    # 🔥 THE HACK: Diver (1) vs Submarine (3)
                    # If one is Diver and the other is Submarine, ALWAYS remove the Submarine.
                    if (cls1 == 1 and cls2 == 3):
                        remove_indices.add(j) # Remove Submarine (at j)
                        continue
                    elif (cls1 == 3 and cls2 == 1):
                        remove_indices.add(i) # Remove Submarine (at i)
                        break # 'i' is gone, stop checking 'j's for it
                    
                    # Standard Logic: Remove lower confidence one
                    remove_indices.add(j)

        # Return only the survivors
        return [d for k, d in enumerate(detections) if k not in remove_indices]

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
