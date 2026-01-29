from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import stream, ws_server
from app.auth import auth_router

# Core Modules
from app.video.video_reader import VideoReader
from app.scheduler.frame_scheduler import FrameScheduler
from app.ml.dummy_ml import DummyML
import threading
import asyncio

app = FastAPI(title="Jal-Drishti Backend", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
# app.include_router(stream.router, prefix="/ws", tags=["stream"]) # Keeping old one for now if needed?
app.include_router(ws_server.router, prefix="/ws", tags=["websocket"])

@app.on_event("startup")
async def startup_event():
    # Capture the main event loop for the WS server to use
    loop = asyncio.get_running_loop()
    ws_server.set_event_loop(loop)
    import os
    # Initialize Core Pipeline
    from app.services.ml_service import ml_service
    
    video_path = "backend/dummy.mp4"
    if not os.path.exists(video_path):
        # Check relative to root as well
        if os.path.exists("dummy.mp4"):
            video_path = "dummy.mp4"
        else:
            print(f"[Startup] Warning: {video_path} not found.")
            return
    
    reader = VideoReader(video_path)

    # Callback to push to WebSocket
    def on_result(envelope):
        """
        Flatten the payload for the frontend.
        The frontend expects 'state', 'image_data', etc. at the top level.
        """
        if envelope.get("type") == "data" and "payload" in envelope:
            flat_payload = {
                "type": "data",
                **envelope["payload"]
            }
            ws_server.broadcast(flat_payload)
        else:
            ws_server.broadcast(envelope)

    # Scheduler
    scheduler = FrameScheduler(reader, target_fps=5, ml_module=ml_service, result_callback=on_result)
    
    # Run in background thread
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()
    print("[Startup] Scheduler thread started with real ML Engine.")


@app.get("/")
def read_root():
    return {"message": "Jal-Drishti Backend is running"}

# Entry point for debugging if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
