"""
Sensor Fusion Module for Jal-Drishti

MILESTONE-1: Layered Sensing Architecture
MILESTONE-3: Multi-Sensor Confidence Fusion & Risk Scoring

This module implements simulated Sonar and IR sensors with rule-based
fusion logic to demonstrate defence-grade multi-sensor detection.

Design Principles:
1. Camera is NOT the primary detector
2. Sensor fusion is logical, not hardware-level
3. Sensor detections persist across frames (no single-frame spikes)
4. Human operator remains in the loop
5. Risk score is weighted and continuous (M3)
"""

import time
import logging
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MILESTONE-4: Late import to avoid circular dependency
# Will be imported at runtime in SensorFusion.__init__


# ============================================================
# SENSOR STATES - Defence-Grade Classification
# ============================================================

class FusionState(Enum):
    """
    Multi-sensor fusion states.
    
    NORMAL: All sensors report clear
    SENSOR_ALERT: Early warning from long-range sensors (Sonar)
    POTENTIAL_ANOMALY: Multiple sensors confirm (Sonar + IR)
    CONFIRMED_THREAT: Full sensor chain confirms (Sonar + IR + Camera)
    SENSOR_DEGRADED: Partial sensor availability (e.g., camera offline)
    """
    NORMAL = "NORMAL"
    SENSOR_ALERT = "SENSOR_ALERT"
    POTENTIAL_ANOMALY = "POTENTIAL_ANOMALY"
    CONFIRMED_THREAT = "CONFIRMED_THREAT"
    SENSOR_DEGRADED = "SENSOR_DEGRADED"


# ============================================================
# CONFIGURATION - Operator-adjustable thresholds
# ============================================================

@dataclass
class FusionConfig:
    """
    Configurable thresholds for sensor fusion.
    
    Camera confirmation threshold is configurable based on 
    operational risk tolerance.
    
    MILESTONE-3: Added sensor weights and risk thresholds.
    """
    # Sonar parameters
    sonar_detection_range_min: float = 10.0  # meters
    sonar_detection_range_max: float = 200.0  # meters
    sonar_confidence_min: float = 0.3
    sonar_confidence_max: float = 0.7
    sonar_lead_frames: int = 36  # ~3 seconds at 12 FPS
    
    # IR parameters
    ir_confidence_min: float = 0.4
    ir_confidence_max: float = 0.8
    ir_delay_frames: int = 24  # ~2 seconds after sonar
    
    # Camera confirmation threshold
    # NOTE: This is configurable based on operational risk tolerance
    camera_confirmation_threshold: float = 0.65
    
    # Persistence settings (avoid single-frame spikes)
    min_detection_frames: int = 3  # Require N consecutive frames
    max_gap_frames: int = 6  # Allow small gaps in detection
    
    # ==== MILESTONE-3: Sensor Weights ====
    # Trust weights for each sensor (sum should approximate 1.0)
    # Sonar: Early detection, but noisy - lower weight
    # IR: Cross-validation sensor - medium weight
    # Camera: Visual certainty - highest weight (but only when supported)
    weight_sonar: float = 0.20
    weight_ir: float = 0.30
    weight_camera: float = 0.50
    
    # ==== MILESTONE-3: Risk Score Thresholds ====
    # Maps continuous risk score [0, 1] to discrete states
    risk_threshold_normal: float = 0.15       # Below = NORMAL
    risk_threshold_sensor_alert: float = 0.35 # Below = SENSOR_ALERT
    risk_threshold_potential: float = 0.60    # Below = POTENTIAL_ANOMALY
    # Above potential threshold = CONFIRMED_THREAT (if sensor chain valid)
    
    # Persistence bonus for stable detections (adds to risk score)
    persistence_bonus: float = 0.10
    
    # Degradation penalty when sensor missing
    degradation_penalty: float = 0.15


# ============================================================
# SENSOR DATA CLASSES
# ============================================================

@dataclass
class SonarReading:
    """Sonar sensor reading."""
    detected: bool = False
    distance_m: float = 0.0
    confidence: float = 0.0
    frame_id: int = 0


@dataclass
class IRReading:
    """IR/Thermal sensor reading."""
    detected: bool = False
    confidence: float = 0.0
    frame_id: int = 0


@dataclass
class CameraReading:
    """Camera/ML sensor reading."""
    detected: bool = False
    confidence: float = 0.0
    frame_id: int = 0
    ml_available: bool = True


@dataclass
class FusedSensorData:
    """Complete fused sensor output with M3 risk score and M4 decision support."""
    sonar: SonarReading
    ir: IRReading
    camera: CameraReading
    fusion_state: FusionState
    fusion_message: str
    timeline_messages: list
    # MILESTONE-3: Continuous risk score
    risk_score: float = 0.0
    # MILESTONE-3: Individual sensor contributions
    sensor_contributions: dict = None
    # MILESTONE-4: Decision support data
    threat_priority: str = "LOW"
    signature: str = ""
    explainability: list = None
    seen_before: bool = False
    occurrence_count: int = 0
    
    def __post_init__(self):
        if self.sensor_contributions is None:
            self.sensor_contributions = {
                'sonar': 0.0, 'ir': 0.0, 'camera': 0.0
            }
        if self.explainability is None:
            self.explainability = []


# ============================================================
# SONAR SIMULATOR
# ============================================================

class SonarSimulator:
    """
    Simulates sonar sensor behavior.
    
    Sonar is the only reliable long-range underwater sensor.
    - Detects objects BEFORE camera visibility
    - Independent of camera visibility
    - Outputs range and confidence
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self._detection_start_frame: Optional[int] = None
        self._consecutive_detections: int = 0
        self._last_detection_frame: int = 0
        self._simulated_distance: float = 0.0
        
    def simulate(self, frame_id: int, camera_confidence: float) -> SonarReading:
        """
        Generate sonar reading based on frame progression.
        
        Sonar detects BEFORE camera - typically when camera confidence
        starts rising but is still too low for confirmation.
        """
        # Simulate detection starting before camera confirmation
        # Sonar leads camera by configured frames
        threshold_for_sonar = self.config.camera_confirmation_threshold * 0.3
        
        # Sonar activates when camera starts picking up something faint
        # OR when we're in a detection cycle
        is_detecting = camera_confidence > threshold_for_sonar
        
        if is_detecting:
            if self._detection_start_frame is None:
                self._detection_start_frame = frame_id
                logger.info(f"[Sonar] Detection initiated at frame {frame_id}")
            
            # Calculate persistence
            frames_since_last = frame_id - self._last_detection_frame
            if frames_since_last <= self.config.max_gap_frames:
                self._consecutive_detections += 1
            else:
                self._consecutive_detections = 1
            
            self._last_detection_frame = frame_id
            
            # Only confirm after minimum consecutive frames (persistence)
            if self._consecutive_detections >= self.config.min_detection_frames:
                # Simulate decreasing distance as object approaches
                progress = min(1.0, camera_confidence / self.config.camera_confirmation_threshold)
                self._simulated_distance = (
                    self.config.sonar_detection_range_max - 
                    progress * (self.config.sonar_detection_range_max - self.config.sonar_detection_range_min)
                )
                
                # Confidence increases as object gets closer
                confidence = (
                    self.config.sonar_confidence_min + 
                    progress * (self.config.sonar_confidence_max - self.config.sonar_confidence_min)
                )
                
                return SonarReading(
                    detected=True,
                    distance_m=round(self._simulated_distance, 1),
                    confidence=round(confidence, 2),
                    frame_id=frame_id
                )
        else:
            # Reset detection state if gap is too large
            if frame_id - self._last_detection_frame > self.config.max_gap_frames * 2:
                self._detection_start_frame = None
                self._consecutive_detections = 0
        
        return SonarReading(frame_id=frame_id)
    
    def reset(self):
        """Reset simulator state."""
        self._detection_start_frame = None
        self._consecutive_detections = 0
        self._last_detection_frame = 0
        self._simulated_distance = 0.0


# ============================================================
# IR/THERMAL SIMULATOR
# ============================================================

class IRSimulator:
    """
    Simulates IR/Thermal sensor behavior.
    
    IR bridges the gap between sonar suspicion and visual confirmation.
    - Activates AFTER sonar detection
    - Confidence higher than sonar but lower than camera
    - Detects heat signatures/silhouettes
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self._consecutive_detections: int = 0
        self._last_detection_frame: int = 0
        self._activation_frame: Optional[int] = None
        
    def simulate(self, frame_id: int, sonar_reading: SonarReading, 
                 camera_confidence: float) -> IRReading:
        """
        Generate IR reading based on sonar state and frame progression.
        
        IR activates after sonar confirms, providing mid-range confirmation.
        """
        # IR only activates after sonar has detected
        if not sonar_reading.detected:
            self._consecutive_detections = 0
            self._activation_frame = None
            return IRReading(frame_id=frame_id)
        
        # Track when IR should activate (delay after sonar)
        if self._activation_frame is None:
            self._activation_frame = frame_id
            
        frames_since_sonar = frame_id - self._activation_frame
        
        # IR needs time to warm up after sonar detection
        if frames_since_sonar < self.config.ir_delay_frames // 3:
            return IRReading(frame_id=frame_id)
        
        # IR detection correlates with camera confidence but leads it
        threshold = self.config.camera_confirmation_threshold * 0.5
        is_detecting = camera_confidence > threshold
        
        if is_detecting:
            # Calculate persistence
            frames_since_last = frame_id - self._last_detection_frame
            if frames_since_last <= self.config.max_gap_frames:
                self._consecutive_detections += 1
            else:
                self._consecutive_detections = 1
            
            self._last_detection_frame = frame_id
            
            # Only confirm after minimum consecutive frames
            if self._consecutive_detections >= self.config.min_detection_frames:
                # IR confidence based on camera proximity to threshold
                progress = min(1.0, camera_confidence / self.config.camera_confirmation_threshold)
                confidence = (
                    self.config.ir_confidence_min + 
                    progress * (self.config.ir_confidence_max - self.config.ir_confidence_min)
                )
                
                return IRReading(
                    detected=True,
                    confidence=round(confidence, 2),
                    frame_id=frame_id
                )
        
        return IRReading(frame_id=frame_id)
    
    def reset(self):
        """Reset simulator state."""
        self._consecutive_detections = 0
        self._last_detection_frame = 0
        self._activation_frame = None


# ============================================================
# SENSOR FUSION ENGINE
# ============================================================

class SensorFusion:
    """
    Rule-based multi-sensor fusion engine.
    
    Fusion Rules:
    - Sonar only → SENSOR_ALERT
    - Sonar + IR → POTENTIAL_ANOMALY  
    - Sonar + IR + Camera(>threshold) → CONFIRMED_THREAT
    - Camera alone CANNOT produce CONFIRMED_THREAT
    - Partial sensor availability → SENSOR_DEGRADED
    """
    
    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.sonar = SonarSimulator(self.config)
        self.ir = IRSimulator(self.config)
        self._last_fusion_state = FusionState.NORMAL
        self._state_change_frame: int = 0
        # MILESTONE-3: Track persistence for bonus
        self._consecutive_detection_frames: int = 0
        
        # MILESTONE-4: Decision support integration
        from app.services.decision_support import get_decision_support
        self._decision_support = get_decision_support()
        
        logger.info(f"[SensorFusion] Initialized with camera threshold: "
                   f"{self.config.camera_confirmation_threshold}")
        logger.info(f"[SensorFusion] M3 Weights: Sonar={self.config.weight_sonar}, "
                   f"IR={self.config.weight_ir}, Camera={self.config.weight_camera}")
        logger.info(f"[SensorFusion] M4 Decision Support enabled")
    
    def process_frame(self, frame_id: int, camera_confidence: float,
                      ml_available: bool = True) -> FusedSensorData:
        """
        Process a single frame through all sensors and apply fusion logic.
        
        Args:
            frame_id: Current frame number
            camera_confidence: ML detection confidence (0-1)
            ml_available: Whether ML/camera is operational
            
        Returns:
            FusedSensorData with all sensor readings and fusion state
        """
        timeline_messages = []
        
        # Step 1: Get sonar reading (always available)
        sonar = self.sonar.simulate(frame_id, camera_confidence)
        
        # Step 2: Get IR reading (based on sonar)
        ir = self.ir.simulate(frame_id, sonar, camera_confidence)
        
        # Step 3: Create camera reading
        camera_detected = camera_confidence >= self.config.camera_confirmation_threshold
        camera = CameraReading(
            detected=camera_detected,
            confidence=round(camera_confidence, 2),
            frame_id=frame_id,
            ml_available=ml_available
        )
        
        # ==== MILESTONE-3: Calculate risk score ====
        risk_score, sensor_contributions = self._calculate_risk_score(
            sonar, ir, camera, ml_available
        )
        
        # Step 4: Apply fusion rules (now risk-aware)
        fusion_state, fusion_message = self._apply_fusion_rules(
            sonar, ir, camera, ml_available, risk_score
        )
        
        # Step 5: Generate timeline messages for state changes
        if fusion_state != self._last_fusion_state:
            timeline_messages = self._generate_timeline_messages(
                sonar, ir, camera, fusion_state, risk_score
            )
            self._last_fusion_state = fusion_state
            self._state_change_frame = frame_id
        
        # ==== MILESTONE-4: Get decision support context ====
        threat_context = self._decision_support.process_detection(
            risk_score=risk_score,
            sonar_distance=sonar.distance_m if sonar.detected else 200.0,
            fusion_state=fusion_state.value,
            sensor_contributions=sensor_contributions,
            persistence_bonus=self.config.persistence_bonus * min(1.0, self._consecutive_detection_frames / 10.0),
            degradation_penalty=self.config.degradation_penalty if not ml_available else 0.0
        )
        
        return FusedSensorData(
            sonar=sonar,
            ir=ir,
            camera=camera,
            fusion_state=fusion_state,
            fusion_message=fusion_message,
            timeline_messages=timeline_messages,
            risk_score=round(risk_score, 3),
            sensor_contributions=sensor_contributions,
            # MILESTONE-4 fields
            threat_priority=threat_context.priority.value,
            signature=self._decision_support.memory._generate_signature(
                risk_score, sonar.distance_m if sonar.detected else 200.0, fusion_state.value
            ),
            explainability=threat_context.explainability,
            seen_before=threat_context.seen_before,
            occurrence_count=threat_context.occurrence_count
        )
    
    # ==== MILESTONE-3: Risk Score Calculation ====
    def _calculate_risk_score(self, sonar: SonarReading, ir: IRReading,
                             camera: CameraReading, ml_available: bool
                             ) -> Tuple[float, dict]:
        """
        Calculate weighted composite risk score.
        
        MILESTONE-3: Smooth risk escalation using sensor weights.
        
        Returns:
            Tuple of (risk_score, sensor_contributions dict)
        """
        contributions = {'sonar': 0.0, 'ir': 0.0, 'camera': 0.0}
        
        # Normalize and weight each sensor's contribution
        # Sonar contribution (normalized to 0-1)
        if sonar.detected:
            # Normalize sonar confidence to full range
            sonar_normalized = (sonar.confidence - self.config.sonar_confidence_min) / (
                self.config.sonar_confidence_max - self.config.sonar_confidence_min
            )
            sonar_normalized = max(0.0, min(1.0, sonar_normalized))
            contributions['sonar'] = round(sonar_normalized * self.config.weight_sonar, 3)
        
        # IR contribution
        if ir.detected:
            ir_normalized = (ir.confidence - self.config.ir_confidence_min) / (
                self.config.ir_confidence_max - self.config.ir_confidence_min
            )
            ir_normalized = max(0.0, min(1.0, ir_normalized))
            contributions['ir'] = round(ir_normalized * self.config.weight_ir, 3)
        
        # Camera contribution (ONLY counts if sensor chain is valid)
        # Camera alone should not inflate risk significantly
        if camera.detected and ml_available:
            if sonar.detected:  # Camera only counts when sonar has detected
                camera_normalized = camera.confidence  # Already 0-1
                contributions['camera'] = round(camera_normalized * self.config.weight_camera, 3)
            else:
                # Camera alone gets reduced weight (cannot confirm alone)
                contributions['camera'] = round(camera.confidence * self.config.weight_camera * 0.3, 3)
        
        # Base risk from sensor contributions
        base_risk = contributions['sonar'] + contributions['ir'] + contributions['camera']
        
        # Track persistence for bonus
        if sonar.detected or ir.detected:
            self._consecutive_detection_frames += 1
        else:
            self._consecutive_detection_frames = 0
        
        # Persistence bonus (stable detections increase confidence)
        persistence_multiplier = min(1.0, self._consecutive_detection_frames / 10.0)
        persistence_bonus = self.config.persistence_bonus * persistence_multiplier
        
        # Degradation penalty if sensor missing
        degradation_penalty = 0.0
        if not ml_available and (sonar.detected or ir.detected):
            degradation_penalty = self.config.degradation_penalty
        
        # Final risk score
        risk_score = base_risk + persistence_bonus - degradation_penalty
        risk_score = max(0.0, min(1.0, risk_score))  # Clamp to [0, 1]
        
        return risk_score, contributions
    
    def _apply_fusion_rules(self, sonar: SonarReading, ir: IRReading,
                           camera: CameraReading, ml_available: bool,
                           risk_score: float = 0.0
                           ) -> Tuple[FusionState, str]:
        """
        Apply defence-grade fusion rules with MILESTONE-3 risk awareness.
        
        CRITICAL: Camera alone can NEVER produce CONFIRMED_THREAT
        """
        # Check for sensor degradation first
        if not ml_available:
            if sonar.detected and ir.detected:
                return (FusionState.SENSOR_DEGRADED, 
                       "Operating with limited sensor availability (camera offline)")
            elif sonar.detected:
                return (FusionState.SENSOR_DEGRADED,
                       "Sonar active, camera unavailable")
            else:
                return (FusionState.SENSOR_DEGRADED,
                       "System operating in degraded mode")
        
        # Full fusion chain: Sonar → IR → Camera
        if sonar.detected and ir.detected and camera.detected:
            return (FusionState.CONFIRMED_THREAT,
                   f"THREAT CONFIRMED: All sensors confirm at {sonar.distance_m}m")
        
        # Partial chain: Sonar + IR
        if sonar.detected and ir.detected:
            return (FusionState.POTENTIAL_ANOMALY,
                   f"Potential anomaly at {sonar.distance_m}m - awaiting visual confirmation")
        
        # Sonar only
        if sonar.detected:
            return (FusionState.SENSOR_ALERT,
                   f"Sonar contact at {sonar.distance_m}m - monitoring")
        
        # Camera only - NOT elevated to threat (design principle)
        if camera.detected:
            # Camera alone cannot confirm threat - treat as potential
            return (FusionState.SENSOR_ALERT,
                   "Visual anomaly detected - awaiting sensor confirmation")
        
        return (FusionState.NORMAL, "All sensors clear")
    
    def _generate_timeline_messages(self, sonar: SonarReading, ir: IRReading,
                                   camera: CameraReading, 
                                   new_state: FusionState,
                                   risk_score: float = 0.0) -> list:
        """Generate operator-friendly timeline messages with M3 risk info."""
        messages = []
        
        if new_state == FusionState.SENSOR_ALERT:
            if sonar.detected:
                messages.append(f"Sonar detected object at {sonar.distance_m} meters")
            else:
                messages.append("Visual anomaly requires sensor verification")
            messages.append(f"Risk Level: {risk_score*100:.0f}%")
                
        elif new_state == FusionState.POTENTIAL_ANOMALY:
            messages.append("IR anomaly detected - thermal signature confirmed")
            messages.append(f"Risk Level: {risk_score*100:.0f}%")
            
        elif new_state == FusionState.CONFIRMED_THREAT:
            messages.append("Visual confirmation achieved")
            messages.append(f"THREAT CONFIRMED at {sonar.distance_m}m ")
            messages.append(f"Final Risk: {risk_score*100:.0f}%")
            
        elif new_state == FusionState.SENSOR_DEGRADED:
            messages.append("Operating in degraded sensor mode")
            messages.append(f"Degraded Risk: {risk_score*100:.0f}%")
            
        elif new_state == FusionState.NORMAL:
            messages.append("All sensors returned to normal")
        
        return messages
    
    def reset(self):
        """Reset all sensor states."""
        self.sonar.reset()
        self.ir.reset()
        self._last_fusion_state = FusionState.NORMAL
        self._state_change_frame = 0
        self._consecutive_detection_frames = 0  # M3: Reset persistence
        logger.info("[SensorFusion] All sensors reset")
    
    def get_sensor_status(self) -> dict:
        """Get current sensor status for diagnostics."""
        return {
            "fusion_active": True,
            "sonar_available": True,  # Simulated, always available
            "ir_available": True,  # Simulated, always available
            "camera_threshold": self.config.camera_confirmation_threshold,
            "persistence_frames": self.config.min_detection_frames
        }


# ============================================================
# SINGLETON INSTANCE
# ============================================================

# Global sensor fusion instance
_sensor_fusion_instance: Optional[SensorFusion] = None


def get_sensor_fusion(config: FusionConfig = None) -> SensorFusion:
    """Get or create the singleton SensorFusion instance."""
    global _sensor_fusion_instance
    if _sensor_fusion_instance is None:
        _sensor_fusion_instance = SensorFusion(config)
    return _sensor_fusion_instance
