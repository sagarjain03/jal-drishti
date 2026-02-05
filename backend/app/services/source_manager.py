"""
SourceManager: Runtime Source Switching Service

PHASE-3 CORE: Singleton service managing video source lifecycle.

States: IDLE -> VIDEO_ACTIVE | CAMERA_WAITING -> CAMERA_ACTIVE | ERROR

CRITICAL FIX (v2):
- Scheduler is a SINGLETON, created once at startup
- Sources are attached/detached via scheduler.attach_source()
- ML worker NEVER stops on source switch
- Camera timeout is FRAME-DRIVEN (not wall-clock)
"""

import threading
import time
import socket
import logging
from enum import Enum
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class SourceState(Enum):
    IDLE = "IDLE"
    VIDEO_ACTIVE = "VIDEO_ACTIVE"
    CAMERA_WAITING = "CAMERA_WAITING"
    CAMERA_ACTIVE = "CAMERA_ACTIVE"
    ERROR = "ERROR"


class SourceManager:
    """
    Singleton service for runtime source switching.
    
    CRITICAL DESIGN (v2):
    - Scheduler is created ONCE and reused
    - Use scheduler.attach_source() / scheduler.detach_source()
    - NEVER recreate scheduler or ML worker
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._state = SourceState.IDLE
        self._current_source_type: Optional[str] = None
        self._current_source = None
        self._state_lock = threading.Lock()
        
        # Scheduler reference (singleton, initialized at configure time)
        self._scheduler = None
        
        # Frame-driven timeout tracking
        self._camera_timeout_seconds = 15.0
        self._timeout_active = False
        self._timeout_thread: Optional[threading.Thread] = None
        
        # Last frame timestamp - FRAME-DRIVEN
        self._last_frame_ts: Optional[float] = None
        
        # Config
        self._video_path = "backend/dummy.mp4"
        self._target_fps = 12
        
        # Callbacks (stored for scheduler)
        self._ml_service = None
        self._on_result_callback: Optional[Callable] = None
        self._on_raw_callback: Optional[Callable] = None
        
        logger.info("[SourceManager] Initialized in IDLE state (v2)")
    
    @property
    def state(self) -> SourceState:
        with self._state_lock:
            return self._state
    
    @property
    def source_type(self) -> Optional[str]:
        return self._current_source_type
    
    @property
    def last_frame_ts(self) -> Optional[float]:
        return self._last_frame_ts
    
    def configure(self, 
                  ml_service,
                  on_result_callback: Callable,
                  on_raw_callback: Callable,
                  event_loop=None,
                  video_path: str = "backend/dummy.mp4",
                  target_fps: int = 12):
        """Configure manager and start singleton scheduler."""
        from app.scheduler.frame_scheduler import get_scheduler
        
        self._ml_service = ml_service
        self._on_result_callback = on_result_callback
        self._on_raw_callback = on_raw_callback
        self._video_path = video_path
        self._target_fps = target_fps
        
        # Get/create singleton scheduler
        self._scheduler = get_scheduler()
        self._scheduler.target_fps = target_fps
        self._scheduler.configure(
            ml_module=ml_service,
            result_callback=on_result_callback,
            raw_callback=on_raw_callback
        )
        
        # Start scheduler (runs forever)
        self._scheduler.start()
        
        logger.info("[SourceManager] Configured with singleton scheduler")
    
    def switch_source(self, source_type: str, video_path: str = None) -> dict:
        """
        Switch to a new source. Non-blocking, returns immediately.
        
        Uses scheduler.attach_source() - NO scheduler recreation.
        """
        logger.info(f"[SourceManager] Switch requested: {source_type} (path={video_path})")
        
        if source_type not in ("video", "camera"):
            return {"success": False, "error": "Invalid source type. Use 'video' or 'camera'"}
        
        # Detach current source (does NOT stop scheduler)
        self._detach_current_source()
        
        # Reset counters
        self._last_frame_ts = None
        
        try:
            if source_type == "video":
                return self._attach_video_source(custom_path=video_path)
            else:
                return self._attach_camera_source()
        except Exception as e:
            logger.error(f"[SourceManager] Error switching source: {e}")
            with self._state_lock:
                self._state = SourceState.ERROR
            return {"success": False, "error": str(e), "state": "ERROR"}
    
    def _detach_current_source(self):
        """Detach current source WITHOUT stopping scheduler."""
        logger.info("[SourceManager] Detaching current source...")
        
        # Stop timeout monitoring
        self._timeout_active = False
        
        # Stop source (NOT scheduler!)
        if self._current_source:
            try:
                if hasattr(self._current_source, 'stop'):
                    self._current_source.stop()
            except Exception as e:
                logger.warning(f"[SourceManager] Error stopping source: {e}")
        
        # Detach from scheduler (scheduler keeps running)
        if self._scheduler:
            self._scheduler.detach_source()
        
        self._current_source = None
        self._current_source_type = None
        self._last_frame_ts = None
        
        with self._state_lock:
            self._state = SourceState.IDLE
        
        logger.info("[SourceManager] Source detached (scheduler still running)")
    
    def _attach_video_source(self, custom_path: str = None) -> dict:
        """Attach video file source."""
        from app.video.video_reader import RawVideoSource
        import os
        
        video_path = custom_path if custom_path else self._video_path
        
        if not os.path.exists(video_path):
            for alt_path in ["dummy.mp4", "backend/dummy.mp4"]:
                if os.path.exists(alt_path):
                    video_path = alt_path
                    break
            else:
                with self._state_lock:
                    self._state = SourceState.ERROR
                return {"success": False, "error": f"Video file not found: {video_path}", "state": "ERROR"}
        
        try:
            reader = RawVideoSource(video_path)
            self._current_source = reader
            self._current_source_type = "video"
            
            # Attach to singleton scheduler (NO recreation)
            if self._scheduler:
                self._scheduler.attach_source(reader)
            
            with self._state_lock:
                self._state = SourceState.VIDEO_ACTIVE
            
            logger.info(f"[SourceManager] Video source attached: {video_path}")
            return {"success": True, "state": "VIDEO_ACTIVE", "source": "video"}
            
        except Exception as e:
            with self._state_lock:
                self._state = SourceState.ERROR
            return {"success": False, "error": str(e), "state": "ERROR"}
    
    def _attach_camera_source(self) -> dict:
        """Attach phone camera source."""
        from app.video.phone_source import phone_camera_source
        
        # Reset phone source state
        phone_camera_source.running = True
        phone_camera_source.connected = False
        phone_camera_source._frame_id = 0
        phone_camera_source._last_inject_time = 0.0
        
        # Clear queue
        while not phone_camera_source.frame_queue.empty():
            try:
                phone_camera_source.frame_queue.get_nowait()
            except:
                break
        
        self._current_source = phone_camera_source
        self._current_source_type = "camera"
        
        # Set initial frame timestamp for timeout
        self._last_frame_ts = time.time()
        
        with self._state_lock:
            self._state = SourceState.CAMERA_WAITING
        
        # Attach to singleton scheduler (NO recreation)
        if self._scheduler:
            self._scheduler.attach_source(phone_camera_source)
        
        # Start frame-driven timeout monitor
        self._timeout_active = True
        self._timeout_thread = threading.Thread(
            target=self._monitor_camera_timeout,
            daemon=True
        )
        self._timeout_thread.start()
        
        logger.info("[SourceManager] Camera source attached, waiting for connection...")
        return {"success": True, "state": "CAMERA_WAITING", "source": "camera"}
    
    def _monitor_camera_timeout(self):
        """Frame-driven timeout: only timeout if no frames for 15s."""
        logger.info("[SourceManager] Timeout monitor started (15s frame-driven)")
        
        while self._timeout_active:
            time.sleep(2.0)
            
            with self._state_lock:
                current_state = self._state
            
            # Exit if not in camera mode
            if current_state not in (SourceState.CAMERA_WAITING, SourceState.CAMERA_ACTIVE):
                return
            
            # Check frame-driven timeout
            if self._last_frame_ts is not None:
                time_since_frame = time.time() - self._last_frame_ts
                if time_since_frame > self._camera_timeout_seconds:
                    logger.warning(f"[SourceManager] Camera timeout ({time_since_frame:.1f}s). Going IDLE.")
                    self._timeout_active = False
                    self._detach_current_source()
                    return
        
        logger.info("[SourceManager] Timeout monitor stopped")
    
    def on_frame_received(self):
        """Called by PhoneCameraSource when frame arrives."""
        self._last_frame_ts = time.time()
        
        with self._state_lock:
            if self._state == SourceState.CAMERA_WAITING:
                self._state = SourceState.CAMERA_ACTIVE
                logger.info("[SourceManager] First frame received! State: CAMERA_ACTIVE")
    
    def notify_camera_disconnected(self):
        """Called when phone camera disconnects."""
        logger.info("[SourceManager] Camera disconnected. Going IDLE.")
        self._timeout_active = False
        self._detach_current_source()
    
    def get_status(self) -> dict:
        """Get current source status."""
        return {
            "state": self.state.value,
            "source": self._current_source_type,
            "last_frame_ts": self._last_frame_ts
        }
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("[SourceManager] Shutting down...")
        self._timeout_active = False
        self._detach_current_source()
        if self._scheduler:
            self._scheduler.shutdown()
        logger.info("[SourceManager] Shutdown complete")


def get_lan_ip() -> str:
    """Get the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# Global singleton instance
source_manager = SourceManager()
