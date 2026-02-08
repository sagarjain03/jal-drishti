"""
PHASE-3 CORE: Singleton Frame Scheduler with Source Hot-Swap

CRITICAL DESIGN RULES:
1. Scheduler is created ONCE at startup
2. ML worker is created ONCE at startup
3. Sources are attached/detached WITHOUT destroying scheduler
4. NEVER stop ML worker on source switch
"""

import time
import threading
import queue
from typing import Optional, Callable, Generator
import logging

# MILESTONE-1: Sensor Fusion Integration
from app.services.sensor_fusion import get_sensor_fusion, FusionState
# MILESTONE-4: Decision Support Integration
from app.services.decision_support import get_decision_support

logger = logging.getLogger(__name__)


class FrameScheduler:
    """
    Singleton Frame Scheduler with hot-swappable sources.
    
    Key behaviors:
    - Created once, never destroyed until shutdown
    - ML worker runs continuously
    - Sources can be attached/detached dynamically
    - Callbacks persist across source switches
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, target_fps: int = 12, ml_module=None, 
                 result_callback=None, raw_callback=None):
        """Initialize singleton scheduler (only runs once)."""
        if self._initialized:
            return
        
        self._initialized = True
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.ml_module = ml_module
        self.result_callback = result_callback
        self.raw_callback = raw_callback
        
        # Source management
        self._current_source = None
        self._source_lock = threading.Lock()
        self._source_active = threading.Event()
        
        # Shutdown control
        self._shutdown = threading.Event()
        
        # ========== ML ADMISSION CONTROL ==========
        self.ml_ready = threading.Event()
        self.ml_ready.set()
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # ML worker (runs forever)
        self.ml_worker_thread = None
        self.ml_stop = threading.Event()
        
        # SAFE MODE state
        self.in_safe_mode = False
        self.safe_mode_reason = None
        
        # Last enhanced result for reuse
        self.last_ml_result = None
        
        # Main loop thread
        self._main_thread = None
        
        # MILESTONE-1: Sensor Fusion Engine
        self.sensor_fusion = get_sensor_fusion()
        
        logger.info("[Scheduler] SINGLETON: Initialized (target FPS=%d)", target_fps)
        logger.info("[Scheduler] Sensor fusion enabled")
    
    def configure(self, ml_module=None, result_callback=None, raw_callback=None):
        """Configure callbacks (can be called after init)."""
        if ml_module:
            self.ml_module = ml_module
        if result_callback:
            self.result_callback = result_callback
        if raw_callback:
            self.raw_callback = raw_callback
        logger.info("[Scheduler] Configured with callbacks")
    
    def start(self):
        """Start scheduler and ML worker threads."""
        if self._main_thread and self._main_thread.is_alive():
            logger.warning("[Scheduler] Already running")
            return
        
        self._shutdown.clear()
        self.ml_stop.clear()
        
        # Start ML worker (runs forever)
        if self.ml_module and not (self.ml_worker_thread and self.ml_worker_thread.is_alive()):
            self.ml_worker_thread = threading.Thread(target=self._ml_worker, daemon=True)
            self.ml_worker_thread.start()
            logger.info("[Scheduler] ML worker started (persistent)")
        
        # Start main loop
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        logger.info("[Scheduler] Main loop started")
    
    def attach_source(self, source):
        """
        Attach a new video source. DOES NOT restart scheduler.
        
        Previous source is cleanly detached first.
        """
        with self._source_lock:
            # Stop old source if exists
            if self._current_source:
                if hasattr(self._current_source, 'stop'):
                    try:
                        self._current_source.stop()
                    except Exception as e:
                        logger.warning("[Scheduler] Error stopping old source: %s", e)
                logger.info("[Scheduler] Detached old source")
            
            # Attach new source
            self._current_source = source
            self._source_active.set()
            
            # Clear stale ML results from previous source
            with self.latest_frame_lock:
                self.latest_frame = None
            self.last_ml_result = None
            
            logger.info("[Scheduler] Attached new source: %s", type(source).__name__)
    
    def detach_source(self):
        """Detach current source without stopping scheduler."""
        with self._source_lock:
            if self._current_source:
                if hasattr(self._current_source, 'stop'):
                    try:
                        self._current_source.stop()
                    except Exception as e:
                        logger.warning("[Scheduler] Error stopping source: %s", e)
                self._current_source = None
                self._source_active.clear()
                logger.info("[Scheduler] Source detached (scheduler still running)")
    
    def _main_loop(self):
        """Main scheduler loop - waits for source, processes frames."""
        logger.info("[Scheduler] Main loop: Waiting for source...")
        
        while not self._shutdown.is_set():
            # Wait for a source to be attached
            if not self._source_active.wait(timeout=1.0):
                continue
            
            # Get current source
            with self._source_lock:
                source = self._current_source
            
            if source is None:
                continue
            
            logger.info("[Scheduler] Processing frames from source")
            self._process_source(source)
            logger.info("[Scheduler] Source processing ended")
        
        logger.info("[Scheduler] Main loop exited")
    
    def _process_source(self, source):
        """Process frames from a single source until it stops or is detached."""
        frame_count = 0
        last_fps_log = time.time()
        last_frame_time = time.time()
        
        try:
            for frame, frame_id, source_timestamp in source.read():
                # Check shutdown
                if self._shutdown.is_set():
                    break
                
                # Check if source was detached
                with self._source_lock:
                    if self._current_source is not source:
                        logger.info("[Scheduler] Source changed mid-stream, exiting")
                        break
                
                current_time = time.time()
                
                # ========== STEP 1: EMIT RAW FRAME IMMEDIATELY ==========
                if self.raw_callback:
                    try:
                        self.raw_callback(frame, frame_id, current_time)
                    except Exception as e:
                        logger.error("[Scheduler] Raw callback error: %s", e)
                
                # ========== STEP 2: SUBMIT TO ML (ADMISSION CONTROL) ==========
                if self.ml_module:
                    with self.latest_frame_lock:
                        self.latest_frame = (frame, frame_id, current_time)
                    
                    if self.ml_ready.is_set():
                        self.ml_ready.clear()
                    
                    # Emit enhanced frame at scheduler pace
                    if self.result_callback and self.last_ml_result:
                        try:
                            # MILESTONE-1: Get camera confidence for fusion
                            camera_confidence = self.last_ml_result.get("max_confidence", 0.0)
                            ml_available = self.last_ml_result.get("ml_available", True)
                            
                            # Process through sensor fusion
                            fusion_data = self.sensor_fusion.process_frame(
                                frame_id=frame_id,
                                camera_confidence=camera_confidence,
                                ml_available=ml_available
                            )
                            
                            # Determine system state from fusion (not camera alone)
                            fusion_state_str = fusion_data.fusion_state.value
                            
                            enhanced_payload = {
                                "frame_id": frame_id,
                                "timestamp": current_time,
                                "detections": self.last_ml_result.get("detections", []),
                                "max_confidence": self.last_ml_result.get("max_confidence", 0.0),
                                # MILESTONE-1: Use fusion state instead of camera-only state
                                "state": fusion_state_str,
                                "image_data": self.last_ml_result.get("image_data"),
                                "system": {
                                    "fps": self.target_fps,
                                    "latency_ms": self.last_ml_result.get("ml_latency_ms", 0.0),
                                    "ml_fps": self.last_ml_result.get("ml_fps", 0.0),
                                    "ml_available": ml_available
                                },
                                "is_cached": not self.ml_ready.is_set(),
                                # MILESTONE-1: Sensor data for frontend
                                "sensors": {
                                    "sonar": {
                                        "detected": fusion_data.sonar.detected,
                                        "distance_m": fusion_data.sonar.distance_m,
                                        "confidence": fusion_data.sonar.confidence
                                    },
                                    "ir": {
                                        "detected": fusion_data.ir.detected,
                                        "confidence": fusion_data.ir.confidence
                                    },
                                    "camera": {
                                        "detected": fusion_data.camera.detected,
                                        "confidence": fusion_data.camera.confidence,
                                        "ml_available": fusion_data.camera.ml_available
                                    }
                                },
                                "fusion_state": fusion_state_str,
                                "fusion_message": fusion_data.fusion_message,
                                "timeline_messages": fusion_data.timeline_messages,
                                # MILESTONE-3: Risk score and contributions
                                "risk_score": fusion_data.risk_score,
                                "sensor_contributions": fusion_data.sensor_contributions,
                                # MILESTONE-4: Decision support data
                                "threat_priority": fusion_data.threat_priority,
                                "signature": fusion_data.signature,
                                "explainability": fusion_data.explainability,
                                "seen_before": fusion_data.seen_before,
                                "occurrence_count": fusion_data.occurrence_count
                            }
                            
                            self.result_callback({
                                "type": "data",
                                "status": "success",
                                "message": "Enhanced frame (paced)",
                                "payload": enhanced_payload
                            })
                        except Exception as e:
                            logger.error("[Scheduler] Enhanced callback error: %s", e)
                
                # ========== STEP 3: PACE-DRIVEN SLEEP ==========
                frame_count += 1
                elapsed_since_last = current_time - last_frame_time
                sleep_duration = self.frame_interval - elapsed_since_last
                
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                
                last_frame_time = time.time()
                
                # ========== STEP 4: LOG FPS ==========
                now = time.time()
                if now - last_fps_log >= 5.0:  # Log every 5 seconds
                    logger.info("[Scheduler] FPS=%d (source: %s)", frame_count // 5, type(source).__name__)
                    frame_count = 0
                    last_fps_log = now
        
        except Exception as e:
            logger.error("[Scheduler] Source processing error: %s", e)
            import traceback
            traceback.print_exc()
    
    def _ml_worker(self):
        """Background ML worker - runs forever, never stops on source switch."""
        logger.info("[ML Worker] Started (PERSISTENT - never stops on switch)")
        
        while not self.ml_stop.is_set():
            # Wait for frame with short timeout
            if not self.ml_ready.is_set():
                with self.latest_frame_lock:
                    if self.latest_frame is None:
                        self.ml_ready.set()
                        time.sleep(0.01)
                        continue
                    
                    frame, frame_id, timestamp = self.latest_frame
                    self.latest_frame = None
                
                try:
                    ml_start = time.time()
                    result = self.ml_module.run_inference(frame)
                    ml_duration = time.time() - ml_start
                    
                    # Calculate ML FPS
                    ml_fps = 0.0
                    if hasattr(self, '_last_ml_time'):
                        delta = time.time() - self._last_ml_time
                        if delta > 0:
                            ml_fps = 1.0 / delta
                    self._last_ml_time = time.time()
                    
                    # Store result
                    self.last_ml_result = result.copy()
                    self.last_ml_result['frame_id'] = frame_id
                    self.last_ml_result['ml_latency_ms'] = ml_duration * 1000
                    self.last_ml_result['ml_fps'] = ml_fps
                    
                    # Recovery check
                    if self.in_safe_mode and ml_duration < 0.5:
                        self.in_safe_mode = False
                        self.safe_mode_reason = None
                        logger.info("[ML Worker] Recovered from safe mode")
                        
                except Exception as e:
                    logger.error("[ML Worker] Inference error: %s", e)
                    if not self.in_safe_mode:
                        self.in_safe_mode = True
                        self.safe_mode_reason = str(e)
                
                finally:
                    self.ml_ready.set()
            else:
                time.sleep(0.01)
        
        logger.info("[ML Worker] Worker stopped (shutdown)")
    
    def shutdown(self):
        """Full shutdown (only on app exit)."""
        logger.info("[Scheduler] Shutting down...")
        self._shutdown.set()
        self.ml_stop.set()
        self._source_active.set()  # Unblock waiting
        
        if self._main_thread:
            self._main_thread.join(timeout=3.0)
        if self.ml_worker_thread:
            self.ml_worker_thread.join(timeout=2.0)
        
        logger.info("[Scheduler] Shutdown complete")


# Singleton instance
def get_scheduler() -> FrameScheduler:
    """Get or create the singleton scheduler."""
    return FrameScheduler()
