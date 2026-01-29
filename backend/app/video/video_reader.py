import cv2
import traceback

class VideoReader:
    def __init__(self, video_path: str):
        """
        Initialize the VideoReader with a path to a video file.
        
        Args:
            video_path (str): The absolute path to the input video file.
        """
        self.video_path = video_path
        
    def read_video(self):
        """
        Generator function to read video frame by frame.
        
        Yields:
            numpy.ndarray: The BGR frame with shape (H, W, 3) and dtype uint8.
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"[VideoReader] Error: Could not open video file {self.video_path}")
            return

        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[VideoReader] Info: End of video reached or cannot read frame.")
                    break
                
                # Validation Logic
                # Ensure dtype is uint8
                if frame.dtype != 'uint8':
                     frame = frame.astype('uint8')

                # Log basic frame info
                # "Outputs basic validation logs for each frame"
                print(f"[VideoReader] Frame {frame_index}: Shape={frame.shape}, Dtype={frame.dtype}")
                
                yield frame
                
                frame_index += 1
                
        except Exception as e:
            print(f"[VideoReader] Error processing video: {e}")
            traceback.print_exc()
            
        finally:
            cap.release()
            print("[VideoReader] Info: Video capture released.")

# Simple main block for direct testing if run as script
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_reader = VideoReader(sys.argv[1])
        for frame in video_reader.read_video():
            pass # Just consume the generator to trigger logs
    else:
        print("Usage: python video_reader.py <path_to_video>")
