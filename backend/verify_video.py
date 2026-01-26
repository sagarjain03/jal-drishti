import cv2
import numpy as np
import os
from app.video.video_reader import VideoReader

def create_dummy_video(filename="dummy.mp4", width=640, height=480, num_frames=30):
    # Use MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
    
    print(f"Generating dummy video: {filename}")
    for i in range(num_frames):
        # Create a frame with random noise
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Add some text to ensure it's not just static noise if we were watching it
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print("Video generation complete.")
    return filename

def verify_video_reader(video_path):
    print(f"Testing VideoReader with {video_path}...")
    reader = VideoReader(video_path)
    
    count = 0
    for frame in reader.read_video():
        count += 1
        # Checks
        assert frame.dtype == np.uint8, f"Frame dtype mismatch: {frame.dtype}"
        assert len(frame.shape) == 3, f"Frame shape mismatch: {frame.shape}"
        assert frame.shape[2] == 3, "Frame is not 3 channels"
        
    print(f"Successfully processed {count} frames.")
    assert count == 30, f"Expected 30 frames, got {count}"

if __name__ == "__main__":
    video_file = create_dummy_video()
    try:
        verify_video_reader(video_file)
        print("VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
    # finally:
    #     if os.path.exists(video_file):
    #         try:
    #             os.remove(video_file)
    #             print("Dummy video cleaned up.")
    #         except:
    #             pass
