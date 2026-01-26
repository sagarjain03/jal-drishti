import logging
# Configure logging to suppress OpenCV output if possible, but mainly to see our scheduler logs
logging.basicConfig(level=logging.INFO)

from app.scheduler.frame_scheduler import FrameScheduler
from app.video.video_reader import VideoReader
import os
import cv2
import numpy as np
import time

def create_long_dummy_video(filename="long_dummy.mp4", duration_sec=5, fps=30):
    width, height = 640, 480
    num_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))
    
    print(f"Generating {duration_sec}s dummy video...")
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"{i}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
        out.write(frame)
    out.release()
    return filename

def test_scheduler_normal():
    print("\n=== TEST 1: Normal Load (Target 15 FPS, Delay 10ms) ===")
    video_path = "long_dummy.mp4"
    if not os.path.exists(video_path):
        create_long_dummy_video(video_path)
        
    reader = VideoReader(video_path)
    # 10ms processing is well within 66ms interval
    scheduler = FrameScheduler(reader, target_fps=15, simulate_processing_delay=0.01)
    
    start = time.time()
    scheduler.run()
    end = time.time()
    
    print(f"Test 1 Complete. Duration: {end-start:.2f}s")

def test_scheduler_overload():
    print("\n=== TEST 2: Overload (Target 15 FPS, Delay 100ms) ===")
    # 100ms processing > 66ms interval. Should see drops.
    video_path = "long_dummy.mp4"
    reader = VideoReader(video_path)
    scheduler = FrameScheduler(reader, target_fps=15, simulate_processing_delay=0.1)
    
    scheduler.run()
    print("Test 2 Complete. Check logs for 'Status=DROPPED'.")

if __name__ == "__main__":
    try:
        test_scheduler_normal()
        test_scheduler_overload()
    finally:
        if os.path.exists("long_dummy.mp4"):
            # os.remove("long_dummy.mp4") # Keep for user inspection if needed, or delete.
            # Let's delete to keep clean unless requested otherwise.
            # Actually, user might want to see it. I'll delete it to be clean.
            try:
                os.remove("long_dummy.mp4")
            except: 
                pass
