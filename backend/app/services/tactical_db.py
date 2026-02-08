"""
Tactical Database for Jal-Drishti

MILESTONE-2: Unknown Object Handling & Human-in-the-Loop Learning

This module implements the "Tactical Adaptation Layer" - the fast, mission-scoped
override system that allows operators to tag unknown objects for immediate use.

Key Concepts (Dual-Speed Learning):
1. TACTICAL (Fast): SQLite overrides for current mission (~0.05ms lookup)
2. STRATEGIC (Slow): Cropped images harvested for future retraining (hours/days)

Design Principles:
- Human-in-the-loop learning only
- No live model retraining
- No autonomous class creation
- Full auditability

This file is INFRASTRUCTURE for the ML engineer to integrate into YOLO inference.
"""

import sqlite3
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TacticalOverride:
    """Single tactical override entry."""
    track_id: int
    signature: str  # Object signature (bbox hash)
    label: str
    timestamp: str
    operator_id: str
    status: str  # ACTIVE, EXPIRED, REVOKED
    confidence_at_tag: float
    mission_id: str


@dataclass
class RecurringUnknown:
    """Tracks recurring unknown object appearances."""
    signature: str
    first_seen: float
    last_seen: float
    occurrence_count: int
    frame_ids: List[int]
    avg_confidence: float
    is_flagged: bool  # True if occurrence >= threshold


# ============================================================
# TACTICAL DATABASE (SQLite)
# ============================================================

class TacticalDB:
    """
    SQLite-backed Tactical Override Database.
    
    The "Digital Sticky Notes" system:
    - Operator tags an unknown object
    - Override stored in SQLite for ultra-fast lookup (~0.05ms)
    - ML engineer integrates this into YOLO loop to override labels
    
    IMPORTANT: This class provides the INFRASTRUCTURE.
    The ML engineer must call get_override() in the YOLO inference loop.
    
    CRITICAL: Uses ABSOLUTE SHARED path so backend and ML engine use SAME database.
    """
    
    # Shared absolute path for database - ensures backend and ML engine use SAME file
    # This must be accessible from both processes
    SHARED_DB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "mission_data", "tactical.db")
    )
    
    # Fixed mission ID for live override sync - both processes use same ID
    # This avoids mission ID mismatch between backend and ML engine
    SHARED_MISSION_ID = "live_mission"
    
    def __init__(self, db_path: str = None):
        """
        Initialize the tactical database.
        
        Args:
            db_path: Path to SQLite database file (defaults to shared absolute path)
        """
        self._lock = Lock()
        
        # Use shared absolute path to ensure backend and ML engine share same DB
        if db_path is None:
            db_path = self.SHARED_DB_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = os.path.abspath(db_path)
        self._init_db()
        
        # Use shared mission ID for live sync between backend and ML engine
        # This ensures overrides added by backend are visible to ML engine query
        self.current_mission_id = self.SHARED_MISSION_ID
        
        logger.info(f"[TacticalDB] Initialized at SHARED path: {self.db_path}")
        logger.info(f"[TacticalDB] Using SHARED mission ID: {self.current_mission_id}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection (thread-safe)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Tactical overrides table (indexed by track_id for fast lookup)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tactical_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    signature TEXT NOT NULL,
                    label TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    operator_id TEXT DEFAULT 'operator_1',
                    status TEXT DEFAULT 'ACTIVE',
                    confidence_at_tag REAL DEFAULT 0.0,
                    mission_id TEXT NOT NULL,
                    UNIQUE(track_id, mission_id)
                )
            ''')
            
            # Create index for fast lookup
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_track_id 
                ON tactical_overrides(track_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_mission_status 
                ON tactical_overrides(mission_id, status)
            ''')
            
            # Recurring unknowns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recurring_unknowns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature TEXT UNIQUE NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    avg_confidence REAL DEFAULT 0.0,
                    is_flagged BOOLEAN DEFAULT 0,
                    mission_id TEXT NOT NULL
                )
            ''')
            
            # Audit log for all tagging actions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tag_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    track_id INTEGER,
                    label TEXT,
                    timestamp DATETIME NOT NULL,
                    operator_id TEXT,
                    mission_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
    
    # ============================================================
    # OVERRIDE OPERATIONS (For ML Engineer Integration)
    # ============================================================
    
    def add_override(self, track_id: int, label: str, signature: str = "",
                    confidence: float = 0.0, operator_id: str = "operator_1") -> bool:
        """
        Add or update a tactical override.
        
        Called when operator labels an object via the UI.
        
        Args:
            track_id: YOLO track ID for the object
            label: Human-assigned label
            signature: Object signature (bbox hash, optional)
            confidence: YOLO confidence at time of tagging
            operator_id: Who made the tag
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # INSERT OR REPLACE: Updates if track_id+mission already exists
                cursor.execute('''
                    INSERT OR REPLACE INTO tactical_overrides 
                    (track_id, signature, label, timestamp, operator_id, 
                     status, confidence_at_tag, mission_id)
                    VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
                ''', (track_id, signature, label, timestamp, operator_id,
                      confidence, self.current_mission_id))
                
                # Audit log
                cursor.execute('''
                    INSERT INTO tag_audit_log (action, track_id, label, timestamp, operator_id, mission_id)
                    VALUES ('ADD_OVERRIDE', ?, ?, ?, ?, ?)
                ''', (track_id, label, timestamp, operator_id, self.current_mission_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"[TacticalDB] Override added: ID {track_id} → '{label}'")
                return True
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error adding override: {e}")
                return False
    
    def get_override(self, track_id: int) -> Optional[str]:
        """
        Get tactical override label for a track ID.
        
        *** FOR ML ENGINEER: Call this in your YOLO inference loop! ***
        
        Latency: ~0.05ms (600x faster than YOLO inference)
        
        CRITICAL: Executes FRESH SQL query every call - no caching!
        
        Args:
            track_id: YOLO track ID
            
        Returns:
            Override label if exists, None otherwise
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT label FROM tactical_overrides 
                    WHERE track_id = ? AND mission_id = ? AND status = 'ACTIVE'
                ''', (track_id, self.current_mission_id))
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    logger.debug(f"[TacticalDB] OVERRIDE FOUND: track_id={track_id} -> '{result['label']}'")
                    return result['label']
                return None
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error getting override: {e}")
                return None
    
    def get_all_active_overrides(self) -> List[TacticalOverride]:
        """Get all active overrides for current mission."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM tactical_overrides 
                    WHERE mission_id = ? AND status = 'ACTIVE'
                    ORDER BY timestamp DESC
                ''', (self.current_mission_id,))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [
                    TacticalOverride(
                        track_id=row['track_id'],
                        signature=row['signature'],
                        label=row['label'],
                        timestamp=row['timestamp'],
                        operator_id=row['operator_id'],
                        status=row['status'],
                        confidence_at_tag=row['confidence_at_tag'],
                        mission_id=row['mission_id']
                    )
                    for row in rows
                ]
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error getting overrides: {e}")
                return []
    
    def revoke_override(self, track_id: int) -> bool:
        """Revoke an override (operator changed their mind)."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE tactical_overrides 
                    SET status = 'REVOKED'
                    WHERE track_id = ? AND mission_id = ?
                ''', (track_id, self.current_mission_id))
                
                # Audit
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO tag_audit_log (action, track_id, label, timestamp, mission_id)
                    VALUES ('REVOKE_OVERRIDE', ?, NULL, ?, ?)
                ''', (track_id, timestamp, self.current_mission_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"[TacticalDB] Override revoked: ID {track_id}")
                return True
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error revoking override: {e}")
                return False
    
    # ============================================================
    # RECURRING UNKNOWN TRACKING
    # ============================================================
    
    def record_unknown_sighting(self, signature: str, confidence: float) -> RecurringUnknown:
        """
        Record a sighting of an unknown object.
        
        Use bbox hash or similar as signature.
        Flags as RECURRING_UNKNOWN if seen >= threshold times.
        
        Args:
            signature: Unique object signature
            confidence: YOLO confidence
            
        Returns:
            Updated RecurringUnknown entry
        """
        RECURRENCE_THRESHOLD = 3  # Flag after 3 sightings
        
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                now = time.time()
                
                # Check if exists
                cursor.execute('''
                    SELECT * FROM recurring_unknowns WHERE signature = ?
                ''', (signature,))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing
                    new_count = existing['occurrence_count'] + 1
                    new_avg = ((existing['avg_confidence'] * existing['occurrence_count']) + confidence) / new_count
                    is_flagged = new_count >= RECURRENCE_THRESHOLD
                    
                    cursor.execute('''
                        UPDATE recurring_unknowns 
                        SET last_seen = ?, occurrence_count = ?, avg_confidence = ?, is_flagged = ?
                        WHERE signature = ?
                    ''', (now, new_count, new_avg, is_flagged, signature))
                    
                    result = RecurringUnknown(
                        signature=signature,
                        first_seen=existing['first_seen'],
                        last_seen=now,
                        occurrence_count=new_count,
                        frame_ids=[],
                        avg_confidence=new_avg,
                        is_flagged=is_flagged
                    )
                else:
                    # Create new entry
                    cursor.execute('''
                        INSERT INTO recurring_unknowns 
                        (signature, first_seen, last_seen, occurrence_count, avg_confidence, is_flagged, mission_id)
                        VALUES (?, ?, ?, 1, ?, 0, ?)
                    ''', (signature, now, now, confidence, self.current_mission_id))
                    
                    result = RecurringUnknown(
                        signature=signature,
                        first_seen=now,
                        last_seen=now,
                        occurrence_count=1,
                        frame_ids=[],
                        avg_confidence=confidence,
                        is_flagged=False
                    )
                
                conn.commit()
                conn.close()
                
                if result.is_flagged:
                    logger.warning(f"[TacticalDB] RECURRING_UNKNOWN flagged: {signature} ({result.occurrence_count} sightings)")
                
                return result
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error recording unknown: {e}")
                return RecurringUnknown(
                    signature=signature, first_seen=time.time(), last_seen=time.time(),
                    occurrence_count=1, frame_ids=[], avg_confidence=confidence, is_flagged=False
                )
    
    def get_flagged_unknowns(self) -> List[RecurringUnknown]:
        """Get all flagged recurring unknowns."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM recurring_unknowns 
                    WHERE is_flagged = 1 AND mission_id = ?
                ''', (self.current_mission_id,))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [
                    RecurringUnknown(
                        signature=row['signature'],
                        first_seen=row['first_seen'],
                        last_seen=row['last_seen'],
                        occurrence_count=row['occurrence_count'],
                        frame_ids=[],
                        avg_confidence=row['avg_confidence'],
                        is_flagged=True
                    )
                    for row in rows
                ]
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error getting flagged unknowns: {e}")
                return []
    
    # ============================================================
    # MISSION MANAGEMENT
    # ============================================================
    
    def start_new_mission(self, mission_id: Optional[str] = None):
        """Start a new mission (clears active overrides context)."""
        self.current_mission_id = mission_id or f"mission_{int(time.time())}"
        logger.info(f"[TacticalDB] New mission started: {self.current_mission_id}")
    
    def end_mission(self):
        """End current mission (marks overrides as expired)."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE tactical_overrides 
                    SET status = 'EXPIRED'
                    WHERE mission_id = ? AND status = 'ACTIVE'
                ''', (self.current_mission_id,))
                
                conn.commit()
                conn.close()
                
                logger.info(f"[TacticalDB] Mission ended: {self.current_mission_id}")
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error ending mission: {e}")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM tactical_overrides WHERE status = "ACTIVE"')
                active = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM recurring_unknowns WHERE is_flagged = 1')
                flagged = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM tag_audit_log')
                audit = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'active_overrides': active,
                    'flagged_unknowns': flagged,
                    'audit_entries': audit,
                    'current_mission': self.current_mission_id
                }
                
            except Exception as e:
                logger.error(f"[TacticalDB] Error getting stats: {e}")
                return {}


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_tactical_db_instance: Optional[TacticalDB] = None


def get_tactical_db() -> TacticalDB:
    """Get or create the singleton TacticalDB instance."""
    global _tactical_db_instance
    if _tactical_db_instance is None:
        _tactical_db_instance = TacticalDB()
    return _tactical_db_instance
