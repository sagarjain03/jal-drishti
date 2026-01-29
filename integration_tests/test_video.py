import sys
import os
import cv2
import json
import base64
import unittest
import numpy as np
import time

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # jal-drishti/

sys.path.insert(0, os.path.join(root_dir, "backend"))
sys.path.insert(0, os.path.join(root_dir, "ml-engine"))

from app.services.ml_service import ml_service

# --- CONFIGURATION ---
# Input video (ensure this file exists!)
INPUT_VIDEO_PATH = os.path.join(root_dir, "backend", "dummy.mp4")
# Output location
OUTPUT_DIR = os.path.join(root_dir, "ml-engine", "outputs")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "processed_test_video.mp4")

class TestFullVideoPipeline(unittest.TestCase):
    
    def test_video_processing(self):
        print(f"\n[VideoTest] Processing Video: {INPUT_VIDEO_PATH}")
        
        # 1. Initialize Video Capture
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        self.assertTrue(cap.isOpened(), f"Could not open video at {INPUT_VIDEO_PATH}")

        # 2. Get Video Properties (to match output)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 15 # Fallback to 15 if unknown
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VideoTest] Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

        # 3. Initialize Video Writer
        # We resize output to 640x640 because that's what the ML Engine returns
        out_size = (640, 640) 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' for .mp4, 'XVID' for .avi
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, out_size)
        
        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # 4. Simulate WebSocket Payload (Encode Frame to Bytes)
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            # 5. Call Backend Pipeline
            # This runs: Decode -> GAN (Enhance) -> YOLO (Detect) -> JSON Response
            resp = ml_service.process_frame(image_bytes)

            # 6. Decode Response Image (The "Enhanced" View)
            # The backend returns the enhanced image as a Base64 string
            if resp["image_data"]:
                b64_str = resp["image_data"]
                img_data = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_data, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Fallback if safe mode returns no image (rare in this flow)
                processed_frame = cv2.resize(frame, out_size)

            # 7. Draw Bounding Boxes (Simulate Frontend Overlay)
            # The backend gives us coordinates; we must draw them to "see" the result in the video.
            detections = resp.get("detections", [])
            for det in detections:
                bbox = det.get("bbox")
                label = det.get("label", "anomaly")
                conf = det.get("confidence", 0.0)
                
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # Red Box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Label
                    text = f"{label} {conf:.2f}"
                    cv2.putText(processed_frame, text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 8. Write to Output Video
            out.write(processed_frame)
            
            # Progress Log
            if frame_idx % 10 == 0:
                print(f" > Frame {frame_idx}/{total_frames} | State: {resp.get('state')} | Detections: {len(detections)}")
            
            frame_idx += 1

        # Cleanup
        cap.release()
        out.release()
        end_time = time.time()
        
        print(f"\n[VideoTest] Processing Complete!")
        print(f"Time Taken: {end_time - start_time:.2f}s")
        print(f"Output saved to: {OUTPUT_VIDEO_PATH}")
        self.assertTrue(os.path.exists(OUTPUT_VIDEO_PATH), "Output video file was not created")

if __name__ == "__main__":
    unittest.main()