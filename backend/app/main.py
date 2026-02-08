import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import stream, ws_server, phone_upload, source_api, viewer_api, operator_api
from app.auth import auth_router

# Core Modules
from app.services.video_stream_manager import video_stream_manager
from app.services.source_manager import source_manager
from app.config_loader import config

import threading
import asyncio
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jal-Drishti Backend", version="2.0.0")

# Mount static files directory for phone_camera.html
_app_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_app_dir)
static_dir = os.path.join(_backend_dir, "static")
print(f"[Main] Static directory: {static_dir}, exists: {os.path.exists(static_dir)}")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print("[Main] Static files mounted at /static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ws_server.router, prefix="/ws", tags=["websocket"])
app.include_router(phone_upload.router, prefix="/ws", tags=["phone"])
app.include_router(source_api.router, tags=["source"])
app.include_router(viewer_api.router, tags=["viewers"])
# MILESTONE-2 & 4: Operator API
app.include_router(operator_api.router)

# --- RAW FEED WEBSOCKET ---
@app.websocket("/ws/raw_feed")
async def raw_feed_endpoint(websocket: WebSocket):
    await video_stream_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        video_stream_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WS] Error in raw_feed: {e}")
        video_stream_manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """
    PHASE-3 CORE: Start in IDLE mode.
    
    No source is attached at startup. User selects source via API.
    ML engine is warmed up and ready.
    """
    # Load configuration
    config.print_summary()
    if not config.validate():
        logger.warning("[Startup] Configuration has validation errors")
    
    # Capture the main event loop
    loop = asyncio.get_running_loop()
    ws_server.set_event_loop(loop)
    
    # Initialize ML Service
    from app.services.ml_service import MLService
    debug_mode = config.get("ml_service.debug_mode", True)
    ml_service = MLService(debug_mode=debug_mode)
    app.state.ml_service = ml_service
    
    # Warm up ML engine in background
    def _warmup_ml():
        try:
            info = ml_service.probe()
            if info and info.get('device'):
                logger.info(f"[Startup] ML engine device: {info.get('device')}")
            # Warmup inference
            import numpy as np
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            ml_service.run_inference(black)
            logger.info("[Startup] ML engine warmup completed")
        except Exception as e:
            logger.debug(f"[Startup] ML warmup skipped: {e}")
    
    warmup_thread = threading.Thread(target=_warmup_ml, daemon=True)
    warmup_thread.start()
    
    # Wait up to 7s for ML warmup
    warmup_start = time.time()
    while time.time() - warmup_start < 7.0:
        if getattr(ml_service, 'engine', None) is not None:
            logger.info("[Startup] ML engine ready")
            break
        time.sleep(0.1)
    
    # Configure SourceManager with callbacks
    def on_result(envelope):
        """Broadcast enhanced frames to WS clients."""
        if envelope.get("type") == "data" and "payload" in envelope:
            flat_payload = {"type": "data", **envelope["payload"]}
            ws_server.broadcast(flat_payload)
        else:
            ws_server.broadcast(envelope)
    
    def raw_frame_callback(frame, frame_id, timestamp):
        """Broadcast raw frames."""
        try:
            if video_stream_manager.is_shutting_down:
                return
            asyncio.run_coroutine_threadsafe(
                video_stream_manager.broadcast_raw_frame(frame, frame_id, timestamp),
                loop
            )
        except Exception as e:
            if not video_stream_manager.is_shutting_down:
                logger.error(f"[Raw] Broadcast error: {e}")
    
    video_path = config.get("video.file_path", "backend/dummy.mp4")
    target_fps = config.get("performance.target_fps", 12)
    
    source_manager.configure(
        ml_service=ml_service,
        on_result_callback=on_result,
        on_raw_callback=raw_frame_callback,
        event_loop=loop,
        video_path=video_path,
        target_fps=target_fps
    )
    
    # Store reference
    app.state.source_manager = source_manager
    
    logger.info("=" * 50)
    logger.info("[Startup] PHASE-3 CORE: Backend started in IDLE mode")
    logger.info("[Startup] Use /api/source/select to start streaming")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown."""
    video_stream_manager.is_shutting_down = True
    
    # Shutdown SourceManager
    if hasattr(app.state, 'source_manager'):
        app.state.source_manager.shutdown()
    
    logger.info("[Shutdown] Backend shutdown complete")


@app.get("/")
def read_root():
    return {
        "message": "Jal-Drishti Backend is running",
        "version": "2.0.0",
        "mode": "IDLE - Use /api/source/select to start streaming"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    status = source_manager.get_status()
    return {
        "status": "healthy",
        "source_state": status["state"],
        "source_type": status["source"]
    }


# Entry point for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)

