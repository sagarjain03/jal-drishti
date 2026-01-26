import time
from app.video.video_reader import VideoReader

class FrameScheduler:
    def __init__(self, video_reader: VideoReader, target_fps: int = 15, simulate_processing_delay: float = 0.0, ml_module = None):
        """
        Initializes the FrameScheduler.

        Args:
            video_reader (VideoReader): Instance of VideoReader source.
            target_fps (int): Target frames per second. Default 15.
            simulate_processing_delay (float): Artificial delay in seconds if no ML module is present.
            ml_module (object): Optional ML module with run_inference method.
        """
        self.video_reader = video_reader
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.simulate_processing_delay = simulate_processing_delay
        self.ml_module = ml_module

    def run(self):
        """
        Runs the scheduler loop.
        Reads frames from the video reader, enforces FPS, and handles logical drops.
        """
        print(f"[Scheduler] Starting at Target FPS={self.target_fps}, Interval={self.frame_interval:.4f}s")
        if self.ml_module:
            print("[Scheduler] ML Module detected. Using synchronous inference.")
        
        start_time = time.time()
        frames_processed_in_window = 0
        last_fps_log_time = start_time

        # We iterate through all frames from the reader
        # logical_frame_index tracks the index in the video source
        for frame in self.video_reader.read_video():
            current_time = time.time()
            elapsed_total = current_time - start_time
            
            if not hasattr(self, '_internal_frame_counter'):
                self._internal_frame_counter = 0
            
            expected_time_for_frame = self._internal_frame_counter * self.frame_interval
            self._internal_frame_counter += 1

            # Check if we are lagging behind real-time
            
            # Reset start_time on first frame to align
            if self._internal_frame_counter == 1:
                start_time = time.time()
                elapsed_total = 0
                expected_time_for_frame = 0

            # Drift check
            drift = elapsed_total - expected_time_for_frame 
            
            if drift > self.frame_interval:
                # We are behind by more than one frame. Drop this frame.
                print(f"[Scheduler] Frame {self._internal_frame_counter-1}: Status=DROPPED (Drift={drift:.4f}s)")
            else:
                # Process the frame
                print(f"[Scheduler] Frame {self._internal_frame_counter-1}: Status=PROCESSED")
                
                # PROCESSING LOGIC
                if self.ml_module:
                    try:
                        # Blocking call to ML
                        # "Scheduler waits for inference result before moving to next frame"
                        # "DROPPED frames are never sent to ML" (handled by else branch)
                        print(f"[Scheduler] Frame {self._internal_frame_counter-1}: Sending to ML...")
                        result = self.ml_module.run_inference(frame)
                        print(f"[Scheduler] Frame {self._internal_frame_counter-1}: ML Result Received.")
                    except Exception as e:
                        print(f"[Scheduler] Frame {self._internal_frame_counter-1}: ML Error: {e}")
                elif self.simulate_processing_delay > 0:
                    # Fallback simulation if no ML module
                    time.sleep(self.simulate_processing_delay)
                
                frames_processed_in_window += 1
                
                # Check if we processed too fast, need to sleep to maintain FPS
                post_process_time = time.time()
                post_elapsed = post_process_time - start_time
                next_frame_expected_time = (self._internal_frame_counter) * self.frame_interval
                
                sleep_duration = next_frame_expected_time - post_elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # FPS Calculation (every 1 second)
            now = time.time()
            if now - last_fps_log_time >= 1.0:
                print(f"[Scheduler] Actual FPS: {frames_processed_in_window}")
                frames_processed_in_window = 0
                last_fps_log_time = now

if __name__ == "__main__":
    # Test Block
    from app.video.video_reader import VideoReader
    import sys
    
    video_path = "dummy.mp4" # Default for quick test if exists
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
    print(f"Testing Scheduler with {video_path}")
    reader = VideoReader(video_path)
    scheduler = FrameScheduler(reader, target_fps=15, simulate_processing_delay=0.02) # 20ms processing
    scheduler.run()
