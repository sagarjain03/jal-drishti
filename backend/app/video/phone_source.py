"""
PhoneCameraSource: Alternative video source that receives frames from phone camera via WebSocket.

This module provides a video source class with the SAME interface as RawVideoSource,
allowing it to be used as a drop-in replacement for the FrameScheduler.

Integration Notes:
- FrameScheduler expects a video source with a read() generator that yields (frame, frame_id, timestamp)
- This class uses a thread-safe queue to receive frames from the WebSocket upload handler
- The scheduler thread calls read() which blocks on the queue, just like RawVideoSource blocks on cv2.VideoCapture
- NO CHANGES to FrameScheduler, ML Bridge, or SAFE MODE logic are required
"""

import queue
import time
import numpy as np
from typing import Tuple, Generator


class PhoneCameraSource:
    """
    Video source that receives frames from a phone camera via WebSocket.
    
    Implements the same interface as RawVideoSource:
    - read() generator yielding (frame, frame_id, timestamp) tuples
    
    PHASE-3 FIX: FPS throttling to prevent queue overflow.
    
    Usage:
        # Create global instance
        phone_camera_source = PhoneCameraSource()
        
        # WebSocket handler calls this when frame arrives:
        phone_camera_source.inject_frame(rgb_frame)
        
        # FrameScheduler consumes frames via:
        for frame, frame_id, timestamp in phone_camera_source.read():
            ...
    """
    
    # FPS throttling settings
    TARGET_CAMERA_FPS = 12
    MIN_FRAME_INTERVAL = 1.0 / TARGET_CAMERA_FPS
    
    def __init__(self, timeout: float = 5.0):
        """
        Initialize the PhoneCameraSource.
        
        Args:
            timeout (float): Seconds to wait for a frame before yielding control.
                            If no frame arrives within timeout, the generator
                            can be used to check for graceful shutdown.
        """
        # Thread-safe queue for frames from WebSocket
        # PHASE-3 FIX: maxsize=30 to handle camera bursts
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=30)
        
        self.timeout = timeout
        self.running = False
        self.connected = False
        self._frame_id = 0
        
        # FPS throttling state
        self._last_inject_time = 0.0
        self._frames_received = 0
        self._frames_dropped_throttle = 0
        
        print("[PhoneCameraSource] Initialized with FPS throttle = 12")
    
    def inject_frame(self, frame: np.ndarray) -> bool:
        """
        Called by the WebSocket upload handler to push a new frame.
        
        PHASE-3 FIX: FPS throttling to prevent queue overflow.
        
        Args:
            frame (np.ndarray): RGB frame with shape (H, W, 3) and dtype uint8
            
        Returns:
            bool: True if frame was accepted, False if throttled or queue full
        """
        if frame is None:
            return False
        
        self._frames_received += 1
        
        # FPS throttling - drop frames that arrive too fast
        now = time.time()
        time_since_last = now - self._last_inject_time
        
        if time_since_last < self.MIN_FRAME_INTERVAL:
            self._frames_dropped_throttle += 1
            # Log every 100 throttled frames
            if self._frames_dropped_throttle % 100 == 0:
                print(f"[PhoneCameraSource] Throttled {self._frames_dropped_throttle} frames (target: {self.TARGET_CAMERA_FPS} FPS)")
            return False
        
        self._last_inject_time = now
        
        try:
            # Non-blocking put with timeout to prevent deadlock
            self.frame_queue.put(frame, block=False)
            
            if not self.connected:
                self.connected = True
                print("[PhoneCameraSource] Phone connected! Receiving frames.")
            
            # CRITICAL: Notify source_manager that we received a frame
            # This enables frame-driven timeout (not wall-clock)
            try:
                from app.services.source_manager import source_manager
                source_manager.on_frame_received()
            except Exception:
                pass  # Don't fail frame injection on source_manager error
            
            return True
            
        except queue.Full:
            # Queue full - scheduler is behind, drop this frame
            if self._frames_received % 50 == 0:
                print("[PhoneCameraSource] Queue full, dropping frame (scheduler behind)")
            return False
    
    def stop(self):
        """Signal the source to stop. Called when phone disconnects."""
        self.running = False
        self.connected = False
        print("[PhoneCameraSource] Stopping...")
        
        # Put a None sentinel to unblock any waiting read()
        try:
            self.frame_queue.put(None, block=False)
        except queue.Full:
            pass
    
    def read(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator function to read frames from the phone camera queue.
        
        This method has the SAME interface as RawVideoSource.read():
        - Yields (frame, frame_id, timestamp) tuples
        - Blocks until a frame is available
        - Returns when stopped or on error
        
        The FrameScheduler consumes this generator identically to how it
        consumes RawVideoSource.read().
        
        Yields:
            tuple: (frame, frame_id, timestamp)
                - frame: np.ndarray with shape (H, W, 3), dtype uint8, RGB format
                - frame_id: int, sequential frame counter
                - timestamp: float, time.time() when frame was received
        """
        self.running = True
        self._frame_id = 0
        
        print("[PhoneCameraSource] Starting read loop. Waiting for frames...")
        
        try:
            while self.running:
                try:
                    # Block waiting for frame with timeout
                    # Timeout allows periodic check of self.running flag
                    frame = self.frame_queue.get(timeout=self.timeout)
                    
                    # Check for stop sentinel
                    if frame is None:
                        print("[PhoneCameraSource] Received stop sentinel.")
                        break
                    
                    # Validate frame
                    if not isinstance(frame, np.ndarray):
                        print(f"[PhoneCameraSource] Warning: Invalid frame type {type(frame)}, skipping")
                        continue
                    
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    timestamp = time.time()
                    
                    # Log periodically (every 30 frames to reduce spam)
                    if self._frame_id % 30 == 0:
                        print(f"[PhoneCameraSource] Frame {self._frame_id}: Shape={frame.shape}, TS={timestamp:.2f}")
                    
                    yield frame, self._frame_id, timestamp
                    
                    self._frame_id += 1
                    
                except queue.Empty:
                    # No frame received within timeout
                    if self.connected:
                        # Phone was connected but stopped sending - potential disconnect
                        print("[PhoneCameraSource] No frame received within timeout. Phone may have disconnected.")
                        # We don't break here - let the scheduler continue and SAFE MODE will trigger
                        # if ML starts failing due to stale/no frames
                        self.connected = False
                    # Continue waiting - phone might reconnect
                    continue
                    
        except Exception as e:
            print(f"[PhoneCameraSource] Error in read loop: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.running = False
            print("[PhoneCameraSource] Read loop ended.")


# Global singleton instance - used by both main.py and phone_upload.py
# This allows the WebSocket handler to inject frames that the scheduler consumes
phone_camera_source = PhoneCameraSource()
