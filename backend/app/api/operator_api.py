"""
Operator API Endpoints for Jal-Drishti

MILESTONE-2 & MILESTONE-4: Operator Decision Support & Unknown Object Handling

Provides endpoints for:
- Operator tagging of unknown objects
- Tactical override management
- Training data harvesting stats
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

from app.services.tactical_db import get_tactical_db, TacticalOverride
from app.services.training_harvester import get_training_harvester
from app.services.decision_support import get_decision_support, OperatorAction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/operator", tags=["operator"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class TagRequest(BaseModel):
    """Request to tag an unknown object."""
    track_id: int
    label: str
    signature: Optional[str] = ""
    confidence: Optional[float] = 0.0
    operator_id: Optional[str] = "operator_1"
    # Optional: image data for immediate harvesting
    image_data: Optional[str] = None
    bbox: Optional[List[int]] = None


class DecisionRequest(BaseModel):
    """Request to record an operator decision (M4)."""
    signature: str
    action: str  # CONFIRM_THREAT, DISMISS_ALERT, MARK_UNKNOWN, MONITOR_ONLY
    risk_score: float
    threat_priority: str


class TagResponse(BaseModel):
    """Response after tagging."""
    success: bool
    message: str
    track_id: int
    label: str


class StatsResponse(BaseModel):
    """Statistics response."""
    tactical_db: Dict
    harvester: Dict
    decision_support: Dict


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/tag", response_model=TagResponse)
async def tag_object(request: TagRequest):
    """
    Tag an unknown object with a human-assigned label.
    
    This creates a tactical override for immediate use AND
    optionally harvests the image for future retraining.
    
    *** FOR ML ENGINEER ***
    After this endpoint is called, your YOLO loop should check
    get_tactical_db().get_override(track_id) and use the human label
    if it exists.
    """
    try:
        db = get_tactical_db()
        harvester = get_training_harvester()
        
        # Add tactical override
        success = db.add_override(
            track_id=request.track_id,
            label=request.label,
            signature=request.signature or "",
            confidence=request.confidence,
            operator_id=request.operator_id
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add override")
        
        # Harvest image if provided
        if request.image_data and request.bbox:
            harvester.harvest_sample(
                image_data=request.image_data,
                label=request.label,
                track_id=request.track_id,
                bbox=request.bbox,
                original_confidence=request.confidence,
                operator_id=request.operator_id,
                mission_id=db.current_mission_id
            )
        
        logger.info(f"[OperatorAPI] Tagged object {request.track_id} as '{request.label}'")
        
        return TagResponse(
            success=True,
            message=f"Object tagged successfully. Override active for mission.",
            track_id=request.track_id,
            label=request.label
        )
        
    except Exception as e:
        logger.error(f"[OperatorAPI] Tag error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision")
async def record_decision(request: DecisionRequest):
    """
    Record an operator decision (Milestone-4).
    
    Logs the decision with full context for audit trail.
    """
    try:
        ds = get_decision_support()
        
        # Map action string to enum
        try:
            action = OperatorAction[request.action]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        # Get priority enum (for type safety)
        from app.services.decision_support import ThreatPriority
        try:
            priority = ThreatPriority[request.threat_priority]
        except KeyError:
            priority = ThreatPriority.LOW
        
        # Record decision
        audit_entry = ds.record_operator_decision(
            signature=request.signature,
            action=action,
            sensor_states={},  # Would come from current frame
            risk_score=request.risk_score,
            threat_priority=priority
        )
        
        logger.info(f"[OperatorAPI] Decision recorded: {request.action}")
        
        return {
            "success": True,
            "message": f"Decision '{request.action}' recorded",
            "timestamp": audit_entry.timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OperatorAPI] Decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tag/{track_id}")
async def revoke_tag(track_id: int):
    """Revoke a tactical override."""
    try:
        db = get_tactical_db()
        success = db.revoke_override(track_id)
        
        if success:
            return {"success": True, "message": f"Override for {track_id} revoked"}
        else:
            raise HTTPException(status_code=404, detail="Override not found")
            
    except Exception as e:
        logger.error(f"[OperatorAPI] Revoke error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overrides")
async def get_active_overrides():
    """Get all active tactical overrides."""
    db = get_tactical_db()
    overrides = db.get_all_active_overrides()
    
    return {
        "count": len(overrides),
        "overrides": [
            {
                "track_id": o.track_id,
                "label": o.label,
                "timestamp": o.timestamp,
                "operator": o.operator_id
            }
            for o in overrides
        ]
    }


@router.get("/unknowns")
async def get_flagged_unknowns():
    """Get all flagged recurring unknowns."""
    db = get_tactical_db()
    unknowns = db.get_flagged_unknowns()
    
    return {
        "count": len(unknowns),
        "unknowns": [
            {
                "signature": u.signature,
                "occurrences": u.occurrence_count,
                "avg_confidence": round(u.avg_confidence, 2),
                "is_flagged": u.is_flagged
            }
            for u in unknowns
        ]
    }


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get comprehensive statistics."""
    db = get_tactical_db()
    harvester = get_training_harvester()
    ds = get_decision_support()
    
    return StatsResponse(
        tactical_db=db.get_stats(),
        harvester=harvester.get_stats(),
        decision_support={
            "pending_alerts": len(ds._pending_alerts),
            "audit_entries": len(ds.audit._log)
        }
    )


@router.post("/mission/start")
async def start_mission(mission_id: Optional[str] = None):
    """Start a new mission."""
    db = get_tactical_db()
    db.start_new_mission(mission_id)
    
    return {
        "success": True,
        "mission_id": db.current_mission_id
    }


@router.post("/mission/end")
async def end_mission():
    """End current mission."""
    db = get_tactical_db()
    mission_id = db.current_mission_id
    db.end_mission()
    
    return {
        "success": True,
        "ended_mission": mission_id
    }
