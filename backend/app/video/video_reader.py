import cv2
import traceback
import os
import time
from typing import Optional, Tuple, Generator
import numpy as np

class RawVideoSource:
    def __init__(self, video_path: Optional[str] = None):
        """
        Initialize the RawVideoSource.
        
        Args:
            video_path (str, optional): Path to video file. 
                                      If None, checks VIDEO_SOURCE_PATH env var, 
                                      then defaults to 'backend/dummy.mp4'.
        """
        # 1. Resolve Video Path
        if video_path:
            self.video_path = video_path
        else:
            self.video_path = os.getenv("VIDEO_SOURCE_PATH", "dummy.mp4")
            
        # Robust Path Resolution
        # Try finding the file in common locations if it doesn't exist at first glance
        if not os.path.exists(self.video_path):
             candidates = [
                 self.video_path,
                 os.path.join("backend", self.video_path),
                 os.path.join("..", self.video_path),
                 os.path.join("..", "backend", "dummy.mp4"),
                 "dummy.mp4"
             ]
             found = False
             for cand in candidates:
                 if os.path.exists(cand):
                     self.video_path = cand
                     found = True
                     break
             
             if not found:
                 # Fallback to absolute path check if CWD is messy
                 # Assuming we are in backend/app/video/../../..
                 # But safer to just warn and let OpenCV fail if not found
                 pass
        
        # Ensure path is absolute for OpenCV to be happy
        if os.path.exists(self.video_path):
            self.video_path = os.path.abspath(self.video_path)
        
        # 2. Configure Debugging
        self.debug_mode = os.getenv("ENABLE_RAW_FRAME_DEBUG", "false").lower() == "true"
        self.debug_dir = "debug_frames"
        self.debug_frame_limit = 50  # Only save first 50 frames
        
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[RawVideoSource] Debug mode ENABLED. Saving first {self.debug_frame_limit} frames to {self.debug_dir}/")

    def read(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator function to read video frame by frame.
        
        Yields:
            numpy.ndarray: The BGR frame with shape (H, W, 3) and dtype uint8.
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"[RawVideoSource] Error: Could not open video file {self.video_path}")
            return

        frame_id = 0

        print(f"[RawVideoSource] Streaming from: {self.video_path}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("[VideoReader] Info: End of video reached or cannot read frame.")
                    break

                # Validation Logic
                # Ensure dtype is uint8
                if frame is None:
                    print("[VideoReader] Warning: Received None frame, skipping")
                    continue

                if frame.dtype != 'uint8':
                    frame = frame.astype('uint8')

                timestamp = time.time()

                # Convert BGR (OpenCV default) to RGB for downstream consumers
                # Keep as BGR for consistency with OpenCV / ML Engine
                rgb_frame = frame 

                # Log basic frame info
                # Outputs basic validation logs for each frame
                print(f"[VideoReader] Frame {frame_id}: Shape={frame.shape}, Dtype={frame.dtype}, TS={timestamp}")

                # Yield a tuple (frame, frame_id, timestamp) to callers that expect it
                yield rgb_frame, frame_id, timestamp

                frame_id += 1
                
        except Exception as e:
            print(f"[RawVideoSource] Error processing video: {e}")
            traceback.print_exc()
            
        finally:
            cap.release()
            print("[RawVideoSource] Info: Video capture released.")

# Backward compatibility alias if needed, or we just update main.py
VideoReader = RawVideoSource

if __name__ == "__main__":
    import sys
    # Quick test
    if len(sys.argv) > 1:
        source = RawVideoSource(sys.argv[1])
    else:
        source = RawVideoSource()
        
    print("Testing RawVideoSource...")
    for f, fid, ts in source.read():
        print(f"Frame {fid}: Shape={f.shape}, TS={ts}")
        if fid > 10: break
