from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import Optional

router = APIRouter()

# Global variable to hold the single active connection
active_connection: Optional[WebSocket] = None

# Global Event Loop reference for thread-safe messaging
# We will set this when the server starts in main.py if needed, 
# or we can assume ws handlers run in the main loop.
_event_loop = None

def set_event_loop(loop):
    global _event_loop
    _event_loop = loop

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    global active_connection
    
    # Requirement: Accept exactly one active client at a time
    # IMPORTANT: Check BEFORE accepting to prevent race conditions
    if active_connection is not None:
        # Accept first to be able to send a message
        await websocket.accept()
        print("[WS] Rejecting secondary connection.")
        # Send JSON error before closing
        await websocket.send_json({
            "type": "system",
            "status": "error",
            "message": "Connection limit reached",
            "payload": None
        })
        await websocket.close()
        return

    # Accept and set as active ATOMICALLY
    await websocket.accept()
    active_connection = websocket
    print("[WS] Client connected.")
    
    # Send connection success message
    await websocket.send_json({
        "type": "system",
        "status": "connected",
        "message": "WebSocket connection established",
        "payload": None
    })
    
    try:
        while True:
            # Keep the connection open.
            # Use receive() to handle all message types properly
            message = await websocket.receive()
            
            # Check the message type
            if message["type"] == "websocket.disconnect":
                break
            elif message["type"] == "websocket.receive":
                # Handle text or bytes data if needed
                data = message.get("text") or message.get("bytes")
                # Optional: handle ping/pong or commands
                pass
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected (exception).")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        # ALWAYS reset active_connection when this handler exits
        print("[WS] Client disconnected.")
        active_connection = None


def broadcast(message: dict):
    """
    Sends a dictionary message to the active WebSocket connection.
    This function is intended to be called from the Scheduler thread.
    It uses asyncio.run_coroutine_threadsafe to jump to the main event loop.
    
    Args:
        message (dict): The full JSON-serializable message to send. 
                        If the message does not contain 'type', it will be wrapped in a default data envelope.
    """
    global active_connection, _event_loop
    
    if active_connection is None:
        return # No one to send to
        
    if _event_loop is None:
        # Fallback if loop wasn't set explicitly (though it should be)
        try:
            _event_loop = asyncio.get_event_loop()
        except:
             print("[WS] Error: No event loop found for broadcast.")
             return

    async def _send():
        if active_connection:
            try:
                # Check if message is already formatted (has 'type')
                # If so, send as is.
                # If not, wrap it (backward compatibility or simple usage)
                if isinstance(message, dict) and "type" in message:
                     await active_connection.send_json(message)
                else:
                    # Default wrap
                    await active_connection.send_json({
                        "type": "data",
                        "status": "success",
                        "message": "New frame data",
                        "payload": message
                    })
            except Exception as e:
                 print(f"[WS] Send Error: {e}")

    # Schedule the send on the main loop
    asyncio.run_coroutine_threadsafe(_send(), _event_loop)
