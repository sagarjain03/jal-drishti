import logging
# Configure logging
logging.basicConfig(level=logging.INFO)

from app.scheduler.frame_scheduler import FrameScheduler
from app.video.video_reader import VideoReader
from app.ml.dummy_ml import DummyML
import os
import cv2
import numpy as np

def create_short_dummy_video(filename="ml_test_video.mp4", duration_sec=3, fps=15):
    width, height = 640, 480
    num_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))
    
    print(f"Generating {duration_sec}s dummy video...")
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"ML Test {i}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        out.write(frame)
    out.release()
    return filename

def verify_ml_bridge():
    print("\n=== VERIFYING ML BRIDGE ===")
    video_path = "ml_test_video.mp4"
    if not os.path.exists(video_path):
        create_short_dummy_video(video_path)
        
    reader = VideoReader(video_path)
    ml_module = DummyML()
    
    # Target 15 FPS -> 66ms interval
    # ML inference -> 50-100ms.
    # This means ML is slightly simpler or slower than real-time. 
    # If ML is 100ms, max FPS is 10. Frame drops should occur.
    
    scheduler = FrameScheduler(reader, target_fps=15, ml_module=ml_module)
    
    print("Starting Scheduler with ML Bridge...")
    scheduler.run()
    print("Verification Completed.")

if __name__ == "__main__":
    try:
        verify_ml_bridge()
    finally:
        if os.path.exists("ml_test_video.mp4"):
            try:
                os.remove("ml_test_video.mp4")
            except: 
                pass
