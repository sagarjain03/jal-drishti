# Testing Guide: Combined Backend & Frontend

This guide explains how to verify that the `ml-engine` and `backend` are correctly integrated and communicating with the `frontend`.

## Prerequisites

1. **Jal-Drishti Environment**: Ensure you have installed the requirements for both `backend` and `ml-engine`.
2. **Video File**: Ensure a video file named `dummy.mp4` exists in the `backend/` directory (or update the path in `app/main.py`).
3. **Model Weights**: Ensure `funie_generator.pth` and `yolov8n.pt` are in `ml-engine/weights/`.

## Step 1: Start the Backend

Open a terminal in the `backend/` directory and run:

```powershell
# Set Python path to include root if necessary
$env:PYTHONPATH = ".."
python -m app.main
```

You should see logs indicating:
- "[Core] Initializing JalDrishti Engine..."
- "[ML Service] Engine Initialized Successfully"
- "[Startup] Scheduler thread started."
- "[Scheduler] Actual FPS: ..."

## Step 2: Start the Frontend

Open a new terminal in the `frontend/` directory and run:

```powershell
npm run dev
```

1. Open the URL provided (default: `http://localhost:5173`).
2. Login with any test credentials (if required by your current auth setup).

## Step 3: Verify the Live Dashboard

Once connected, monitor the following in the UI:

1. **Enhanced Feed**: This should show the video frames with a clear underwater appearance (GANS enhancement).
2. **Bounding Boxes**: If objects (trash, fish, etc.) are detected, green/red boxes should appear overlaid on the "Enhanced Feed".
3. **Telemetry Metrics**:
   - **System State**: Should toggle between `SAFE_MODE`, `POTENTIAL_ANOMALY`, and `CONFIRMED_THREAT`.
   - **ML Latency**: Should show the processing time per frame in milliseconds (ms).
   - **ML FPS**: Should show how many frames the ML engine is processing per second.

## Troubleshooting

- **No Video?** Ensure `dummy.mp4` is valid and the path in `app/main.py` is correct.
- **Red-Shifted Video?** This indicates a BGR/RGB mismatch. Ensure `VideoReader` is yielding BGR and `pipeline.py` is converting to RGB only for the GAN inference.
- **Connection Failed?** Check the backend console for an `Internal Server Error` or a WebSocket crash. Ensure the `ws_server.broadcast` loop is running.
- **High Latency?** Ensure you are running with GPU (`cuda`) if available. The logs will indicate `[Core] Using device: cuda` or `cpu`.
