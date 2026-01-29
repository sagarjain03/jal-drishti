from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    bbox: List[int] # [x1, y1, x2, y2]
    confidence: float
    label: str

class AIResponse(BaseModel):
    status: str
    frame_id: int
    image_data: str # base64
    timestamp: str
    state: str
    max_confidence: float
    detections: List[Detection]
    visibility_score: Optional[float] = 0.0

class ErrorResponse(BaseModel):
    status: str
    message: str
