from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from app.services.ml_service import ml_service
from app.schemas.response import AIResponse
from app.core.security import verify_token
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn")

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    # 1. Verify Token
    payload = verify_token(token)
    if not payload:
        # Close with Policy Violation (1008) if invalid
        logger.warning("Unauthenticated WebSocket connection attempt rejected.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = payload.get("sub")
    logger.info(f"Authenticated client '{user}' connected to stream")

    await websocket.accept()
    
    try:
        while True:
            # Receive binary frame (bytes)
            # We use receive_bytes to accept raw image data
            data = await websocket.receive_bytes()
            
            # Process frame (Dummy ML)
            result = ml_service.process_frame(data)
            
            # Validate against schema (optional, but good for safety)
            response = AIResponse(**result)
            
            # Send JSON response
            await websocket.send_json(response.model_dump())
            
    except WebSocketDisconnect:
        logger.info(f"Client '{user}' disconnected")
    except Exception as e:
        logger.error(f"Error in stream: {e}")
        try:
            await websocket.close()
        except:
            pass
