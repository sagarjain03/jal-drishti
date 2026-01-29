import base64
import numpy as np
import cv2
import sys
import os

# Ensure backend can import ml-engine core
# Assumes structure:
# jal-drishti/
#   backend/app/services/ml_service.py
#   ml-engine/core/...
# We need to add jal-drishti/ml-engine to path
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../backend/app/services
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # .../jal-drishti
ml_engine_path = os.path.join(root_dir, "ml-engine")
if ml_engine_path not in sys.path:
    sys.path.append(ml_engine_path)

from core.pipeline import JalDrishtiEngine

class MLService:
    def __init__(self):
        print("[ML Service] Initializing...")
        try:
            self.engine = JalDrishtiEngine()
            print("[ML Service] Engine Initialized Successfully")
        except Exception as e:
            print(f"[ML Service] CRITICAL ERROR: Engine failed to start: {e}")
            self.engine = None
        
        self.frame_count = 0

    def run_inference(self, frame: np.ndarray) -> dict:
        """
        Processes a raw BGR frame from the scheduler.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
            
        result_json, enhanced_frame = self.engine.infer(frame)
        
        # Encode enhanced frame back to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', enhanced_frame)
        b64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Flattened response for frontend
        response = {
            "image_data": b64_image,
            "visibility_score": 0.8, # Derived or placeholder
            **result_json # timestamp, state, max_confidence, detections
        }
        return response

    def process_frame(self, binary_frame: bytes) -> dict:
        """
        Real ML Processing for individual frames (e.g. from WebSocket or HTTP).
        """
        self.frame_count += 1
        
        try:
            # Decode image
            nparr = np.frombuffer(binary_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return self._error_response("Image decode failed")
            
            # Run Inference
            result = self.run_inference(frame)
            
            # Combine with status and frame_id
            return {
                "status": "success",
                "frame_id": self.frame_count,
                **result
            }

        except Exception as e:
            print(f"[ML Service] Processing Error: {e}")
            return self._error_response(str(e))

    def _error_response(self, msg):
        return {
            "status": "error",
            "message": msg,
            "state": "SAFE_MODE",
            "frame_id": self.frame_count,
            "detections": [],
            "max_confidence": 0.0,
            "image_data": None
        }

ml_service = MLService()
