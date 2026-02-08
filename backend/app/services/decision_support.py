"""
Decision Support Module for Jal-Drishti

MILESTONE-4: Operator Decision Support & Auditability Layer

This module implements:
1. Threat prioritization based on risk, distance, persistence
2. Situational memory for context awareness
3. Operator action handling and audit logging
4. Explainability support

Design Principles:
1. AI suggests, operator decides
2. No autonomous execution
3. Every decision is logged
4. Context > raw alerts
5. Explainability over automation
"""

import time
import logging
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# THREAT PRIORITY LEVELS
# ============================================================

class ThreatPriority(Enum):
    """
    Advisory priority levels for operator attention.
    These are RECOMMENDATIONS, not decisions.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OperatorAction(Enum):
    """
    Possible operator decisions.
    AI NEVER changes state after operator decision without new evidence.
    """
    CONFIRM_THREAT = "CONFIRM_THREAT"
    DISMISS_ALERT = "DISMISS_ALERT"
    MARK_UNKNOWN = "MARK_UNKNOWN"
    MONITOR_ONLY = "MONITOR_ONLY"
    PENDING = "PENDING"  # No decision yet


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SituationalMemoryEntry:
    """Single entry in situational memory."""
    signature: str  # Hash of detection characteristics
    first_seen: float  # Unix timestamp
    last_seen: float
    occurrence_count: int = 1
    max_risk_score: float = 0.0
    last_operator_action: OperatorAction = OperatorAction.PENDING
    action_timestamp: Optional[float] = None


@dataclass
class ThreatContext:
    """Full context for a detection, enabling informed decisions."""
    priority: ThreatPriority
    risk_score: float
    seen_before: bool = False
    occurrence_count: int = 0
    first_seen_ago: str = ""  # Human-readable "5 minutes ago"
    last_operator_action: OperatorAction = OperatorAction.PENDING
    explainability: List[str] = field(default_factory=list)


@dataclass
class AuditLogEntry:
    """Complete audit trail entry for a decision."""
    timestamp: str
    timestamp_unix: float
    sensor_states: Dict
    risk_score: float
    threat_priority: str
    operator_action: str
    decision_latency_ms: float  # Time from alert to decision
    signature: str


# ============================================================
# SITUATION MEMORY BUFFER
# ============================================================

class SituationalMemory:
    """
    Short-term memory for context awareness.
    Answers: "Have we seen this before?"
    
    No database required - in-memory with optional export.
    """
    
    def __init__(self, max_entries: int = 50, memory_window_sec: float = 1200.0):
        """
        Args:
            max_entries: Maximum entries to retain
            memory_window_sec: How long to remember (default 20 minutes)
        """
        self.max_entries = max_entries
        self.memory_window_sec = memory_window_sec
        self._memory: Dict[str, SituationalMemoryEntry] = {}
        self._access_order = deque(maxlen=max_entries)
        
        logger.info(f"[SituationalMemory] Initialized with {max_entries} slots, "
                   f"{memory_window_sec}s window")
    
    def _generate_signature(self, risk_score: float, sonar_distance: float,
                           fusion_state: str) -> str:
        """Generate a signature for this detection context."""
        # Quantize to reduce noise
        risk_bucket = round(risk_score, 1)
        distance_bucket = round(sonar_distance / 10) * 10  # 10m buckets
        
        sig_str = f"{fusion_state}_{risk_bucket}_{distance_bucket}"
        return hashlib.md5(sig_str.encode()).hexdigest()[:8]
    
    def record_detection(self, risk_score: float, sonar_distance: float,
                        fusion_state: str) -> Tuple[str, SituationalMemoryEntry]:
        """
        Record a detection and return context.
        
        Returns:
            Tuple of (signature, memory_entry)
        """
        now = time.time()
        signature = self._generate_signature(risk_score, sonar_distance, fusion_state)
        
        # Clean expired entries
        self._cleanup_expired(now)
        
        if signature in self._memory:
            # Update existing entry
            entry = self._memory[signature]
            entry.last_seen = now
            entry.occurrence_count += 1
            entry.max_risk_score = max(entry.max_risk_score, risk_score)
            
            # Move to end of access order
            if signature in self._access_order:
                self._access_order.remove(signature)
            self._access_order.append(signature)
        else:
            # Create new entry
            entry = SituationalMemoryEntry(
                signature=signature,
                first_seen=now,
                last_seen=now,
                occurrence_count=1,
                max_risk_score=risk_score
            )
            self._memory[signature] = entry
            self._access_order.append(signature)
            
            # Evict oldest if over capacity
            if len(self._memory) > self.max_entries:
                oldest_sig = self._access_order.popleft()
                if oldest_sig in self._memory:
                    del self._memory[oldest_sig]
        
        return signature, entry
    
    def get_context(self, signature: str) -> Optional[SituationalMemoryEntry]:
        """Get context for a signature if it exists."""
        return self._memory.get(signature)
    
    def record_operator_action(self, signature: str, action: OperatorAction):
        """Record operator action for a detection."""
        if signature in self._memory:
            self._memory[signature].last_operator_action = action
            self._memory[signature].action_timestamp = time.time()
            logger.info(f"[SituationalMemory] Recorded {action.value} for {signature}")
    
    def _cleanup_expired(self, now: float):
        """Remove entries older than memory window."""
        expired = [
            sig for sig, entry in self._memory.items()
            if now - entry.last_seen > self.memory_window_sec
        ]
        for sig in expired:
            del self._memory[sig]
            if sig in self._access_order:
                self._access_order.remove(sig)
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable 'X ago'."""
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff / 60)}m ago"
        else:
            return f"{int(diff / 3600)}h ago"


# ============================================================
# AUDIT LOG
# ============================================================

class AuditLog:
    """
    Complete decision audit trail.
    In-memory with JSON export capability.
    """
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._log: deque = deque(maxlen=max_entries)
        logger.info(f"[AuditLog] Initialized with {max_entries} entry capacity")
    
    def record_decision(self, sensor_states: Dict, risk_score: float,
                       threat_priority: ThreatPriority, 
                       operator_action: OperatorAction,
                       decision_latency_ms: float,
                       signature: str):
        """Record a complete decision for audit."""
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            timestamp_unix=time.time(),
            sensor_states=sensor_states,
            risk_score=round(risk_score, 3),
            threat_priority=threat_priority.value,
            operator_action=operator_action.value,
            decision_latency_ms=round(decision_latency_ms, 2),
            signature=signature
        )
        self._log.append(entry)
        logger.info(f"[AuditLog] Decision recorded: {operator_action.value} "
                   f"(latency: {decision_latency_ms:.0f}ms)")
        return entry
    
    def get_recent(self, count: int = 10) -> List[AuditLogEntry]:
        """Get most recent audit entries."""
        return list(self._log)[-count:]
    
    def export_json(self) -> str:
        """Export audit log as JSON string."""
        entries = [
            {
                "timestamp": e.timestamp,
                "risk_score": e.risk_score,
                "priority": e.threat_priority,
                "action": e.operator_action,
                "latency_ms": e.decision_latency_ms,
                "signature": e.signature
            }
            for e in self._log
        ]
        return json.dumps(entries, indent=2)


# ============================================================
# THREAT PRIORITIZATION ENGINE
# ============================================================

class ThreatPrioritizer:
    """
    Derives threat priority from existing data.
    Priority is ADVISORY only - does not trigger actions.
    """
    
    @staticmethod
    def calculate_priority(risk_score: float, 
                          sonar_distance: float,
                          occurrence_count: int,
                          sensor_count: int) -> ThreatPriority:
        """
        Calculate threat priority from multiple factors.
        
        Args:
            risk_score: Composite risk from M3 (0-1)
            sonar_distance: Distance in meters (lower = higher priority)
            occurrence_count: How many times seen (higher = higher priority)
            sensor_count: How many sensors detecting (more = higher priority)
        """
        priority_score = 0.0
        
        # Factor 1: Risk score (40% weight)
        priority_score += risk_score * 0.40
        
        # Factor 2: Distance (30% weight) - closer = higher priority
        # Normalize: 200m = 0, 0m = 1
        distance_factor = max(0, 1.0 - (sonar_distance / 200.0))
        priority_score += distance_factor * 0.30
        
        # Factor 3: Persistence (20% weight)
        # Cap at 10 occurrences
        persistence_factor = min(1.0, occurrence_count / 10.0)
        priority_score += persistence_factor * 0.20
        
        # Factor 4: Sensor diversity (10% weight)
        # 3 sensors = full score, 1 sensor = 33%
        diversity_factor = sensor_count / 3.0
        priority_score += diversity_factor * 0.10
        
        # Map to priority levels
        if priority_score < 0.25:
            return ThreatPriority.LOW
        elif priority_score < 0.50:
            return ThreatPriority.MEDIUM
        elif priority_score < 0.75:
            return ThreatPriority.HIGH
        else:
            return ThreatPriority.CRITICAL


# ============================================================
# EXPLAINABILITY ENGINE
# ============================================================

class ExplainabilityEngine:
    """
    Generates human-readable explanations for alerts.
    No math shown - only reasoning.
    """
    
    @staticmethod
    def generate_explanation(sensor_contributions: Dict,
                            persistence_bonus: float,
                            degradation_penalty: float,
                            occurrence_count: int,
                            seen_before: bool) -> List[str]:
        """Generate explanation strings for an alert."""
        explanations = []
        
        # Sensor contributions
        if sensor_contributions.get('sonar', 0) > 0:
            pct = int(sensor_contributions['sonar'] * 100)
            explanations.append(f"Sonar detection contributing {pct}% to risk")
        
        if sensor_contributions.get('ir', 0) > 0:
            pct = int(sensor_contributions['ir'] * 100)
            explanations.append(f"IR/thermal confirmation adding {pct}%")
        
        if sensor_contributions.get('camera', 0) > 0:
            pct = int(sensor_contributions['camera'] * 100)
            explanations.append(f"Visual confirmation adding {pct}%")
        
        # Context factors
        if persistence_bonus > 0:
            explanations.append(f"Stable detection (+{int(persistence_bonus*100)}% confidence)")
        
        if degradation_penalty > 0:
            explanations.append(f"Operating with limited sensors (-{int(degradation_penalty*100)}%)")
        
        if seen_before and occurrence_count > 1:
            explanations.append(f"Similar object seen {occurrence_count} times recently")
        
        return explanations


# ============================================================
# DECISION SUPPORT MANAGER (Main Interface)
# ============================================================

class DecisionSupportManager:
    """
    Main interface for Milestone-4 decision support.
    
    Coordinates threat prioritization, memory, explainability, and audit.
    """
    
    def __init__(self):
        self.memory = SituationalMemory()
        self.audit = AuditLog()
        self.prioritizer = ThreatPrioritizer()
        self.explainer = ExplainabilityEngine()
        
        # Track pending alerts for decision latency calculation
        self._pending_alerts: Dict[str, float] = {}  # signature -> alert_time
        
        logger.info("[DecisionSupport] Initialized M4 Decision Support Manager")
    
    def process_detection(self, risk_score: float,
                         sonar_distance: float,
                         fusion_state: str,
                         sensor_contributions: Dict,
                         persistence_bonus: float = 0.0,
                         degradation_penalty: float = 0.0) -> ThreatContext:
        """
        Process a detection and return full threat context.
        
        This is called every frame with detection data.
        """
        # Count active sensors
        sensor_count = sum([
            1 if sensor_contributions.get('sonar', 0) > 0 else 0,
            1 if sensor_contributions.get('ir', 0) > 0 else 0,
            1 if sensor_contributions.get('camera', 0) > 0 else 0
        ])
        
        # Record in situational memory
        signature, memory_entry = self.memory.record_detection(
            risk_score, sonar_distance, fusion_state
        )
        
        # Track alert timing for latency calculation
        if risk_score > 0.15 and signature not in self._pending_alerts:
            self._pending_alerts[signature] = time.time()
        
        # Calculate priority
        priority = self.prioritizer.calculate_priority(
            risk_score=risk_score,
            sonar_distance=sonar_distance,
            occurrence_count=memory_entry.occurrence_count,
            sensor_count=sensor_count
        )
        
        # Generate explainability
        explanations = self.explainer.generate_explanation(
            sensor_contributions=sensor_contributions,
            persistence_bonus=persistence_bonus,
            degradation_penalty=degradation_penalty,
            occurrence_count=memory_entry.occurrence_count,
            seen_before=memory_entry.occurrence_count > 1
        )
        
        # Build context
        context = ThreatContext(
            priority=priority,
            risk_score=risk_score,
            seen_before=memory_entry.occurrence_count > 1,
            occurrence_count=memory_entry.occurrence_count,
            first_seen_ago=self.memory._format_time_ago(memory_entry.first_seen),
            last_operator_action=memory_entry.last_operator_action,
            explainability=explanations
        )
        
        return context
    
    def record_operator_decision(self, signature: str,
                                action: OperatorAction,
                                sensor_states: Dict,
                                risk_score: float,
                                threat_priority: ThreatPriority) -> AuditLogEntry:
        """
        Record an operator decision with full audit trail.
        
        AI NEVER changes state after this without new evidence.
        """
        # Calculate decision latency
        alert_time = self._pending_alerts.get(signature, time.time())
        decision_latency_ms = (time.time() - alert_time) * 1000
        
        # Update memory with operator action
        self.memory.record_operator_action(signature, action)
        
        # Create audit entry
        audit_entry = self.audit.record_decision(
            sensor_states=sensor_states,
            risk_score=risk_score,
            threat_priority=threat_priority,
            operator_action=action,
            decision_latency_ms=decision_latency_ms,
            signature=signature
        )
        
        # Clear from pending if decision made
        if action != OperatorAction.PENDING:
            self._pending_alerts.pop(signature, None)
        
        return audit_entry
    
    def get_audit_log(self, count: int = 10) -> List[AuditLogEntry]:
        """Get recent audit entries."""
        return self.audit.get_recent(count)
    
    def export_audit_log(self) -> str:
        """Export full audit log as JSON."""
        return self.audit.export_json()


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_decision_support_instance: Optional[DecisionSupportManager] = None


def get_decision_support() -> DecisionSupportManager:
    """Get or create the singleton DecisionSupportManager instance."""
    global _decision_support_instance
    if _decision_support_instance is None:
        _decision_support_instance = DecisionSupportManager()
    return _decision_support_instance
