# Jal-Drishti ML Architecture & Iteration Log
*Living documentation of the Machine Learning Pipeline.*

---

## 1. High-Level Architecture: "Dual-Stream Hybrid Pipeline"

The core philosophy of the Jal-Drishti engine is **Sensor Fusion**. We treat the standard optical feed and the AI-Enhanced feed as two separate "sensors" to maximize detection probability across different conditions.

### The Pipeline Flow
1.  **Input**: Raw underwater frame (RGB).
2.  **Stream A (Raw)**: Fed directly to YOLOv8. Best for detecting **Divers** and objects in **clear water**.
3.  **Stream B (Enhanced)**:
    *   **Step 1 (GAN)**: FUnIE-GAN enhances the image (color correction).
    *   **Step 2 (CLAHE)**: Contrast Limited Adaptive Histogram Equalization recovers texture details.
    *   **Step 3 (YOLO)**: Fed to YOLOv8. Best for detecting **Mines** and objects in **murky/turbid water**.
4.  **Merge & Filter**: Results are combined, filtered by "Smart NMS", and visualized.

---

## 2. Technical Details

### A. Image Enhancement (FUnIE-GAN)
*   **Model**: Fast Underwater Image Enhancement GAN (FUnIE-GAN).
*   **Input Resolution**: Resized to `256x256` (Model's native training size).
*   **Normalization**:
    *   Input: Scaled to `[-1, 1]` range. `(x - 127.5) / 127.5`.
    *   Output: Re-scaled to `[0, 1]`. `(x + 1) / 2`.
*   **Post-Process**:
    *   Upscaled back to original resolution using **Bilinear Interpolation** (PyTorch `F.interpolate`).
    *   Values clamped to `[0, 1]` and converted to `uint8` `[0, 255]`.

### B. Contrast Enhancement (CLAHE)
*   **Start**: Applied *after* GAN enhancement.
*   **Logic**:
    1.  Convert Image to **LAB** Color Space.
    2.  Extract **L-Channel** (Luminance/Lightness).
    3.  Apply CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`).
    4.  Merge channels and convert back to **BGR**.
*   **Why?**: GANs fix color, but sometimes wash out texture. CLAHE brings back the edges needed for YOLO detection.

### C. Object Detection (YOLOv8)
*   **Model**: YOLOv8 (Custom Trained).
*   **Classes**:
    *   `0`: Mine
    *   `1`: Diver
    *   `2`: Drone (ROV/AUV)
    *   `3`: Submarine
*   **Optimization (Batch Inference)**:
    *   Instead of running inference twice (once for Raw, once for Enhanced), we stack images into a batch `[Raw, Enhanced]`.
    *   Run `model.predict(batch)` **once**.
    *   **Benefit**: ~2x FPS improvement.

---

## 3. Intelligent Logic Layers

We don't just trust the model blindly. We wrap it in logic layers to handle edge cases.

### Logic 1: The "Gatekeeper" (Class-Specific Thresholds)
Different objects have different risks of False Positives. We discriminate based on Class ID.

| Class | Object | Threshold | Reasoning |
| :--- | :--- | :--- | :--- |
| `1` | **Diver** | **> 0.55** | **High**. Prevents "Ghost Divers" (fish/noise). We only alert if sure. |
| `2` | **Drone** | **> 0.15** | **Low**. Drones are rare and small. Catches faint signatures. |
| `0` | **Mine** | **> 0.40** | **Balanced**. Standard safety protocol. |
| `3` | **Sub** | **> 0.40** | **Balanced**. Standard safety protocol. |

### Logic 2: Smart NMS (Conflict Resolution)
Standard NMS merges bounding boxes based on overlap. Our **Smart NMS** resolves semantic conflicts.

*   **The "Diver Priority" Rule**:
    *   **Scenario**: Model detects a "Submarine" AND a "Diver" in the exact same spot (High IoU > 0.4).
    *   **Problem**: Big ROVs or noisy backgrounds often look like subs. Detecting a Submarine on top of a Diver is a critical failure.
    *   **Fix**: If Class `1` (Diver) overlaps Class `3` (Submarine), **DELETE THE SUBMARINE**.
    *   **Result**: The system prioritizes the Human presence.

---

## 4. Visual Feedback Strategy

The Dashboard provides immediate visual attribution for *why* something was detected.

*   **Green Box**: `RAW_SENSOR` Detection.
    *   Means: "Detected by standard optics." (Clear water, close range).
*   **Orange Box**: `AI_ENHANCED` Detection.
    *   Means: "Detected thanks to AI Enhancement." (Murky water, hidden threats).
*   **Text**: Black text on colored background for maximum readability.

---

## 5. Iteration Log (The "Why")

### Phase 1: The "Night Vision" Problem
*   **Issue**: Initial GAN outputs were great for machines (high contrast) but "ugly" for humans (red/muddy).
*   **Attempt 1**: Applied a "Cinematic Blue" filter for the UI.
*   **Lesson**: It corrupted the visual data for the user.
*   **Final Fix**: Removed filters. We show the "Scientific Red" output but annotate it cleanly. The user sees exactly what the AI sees.

### Phase 2: The FPS Drop
*   **Issue**: Dual-Stream (Raw + Enhanced) cut FPS in half (15 -> 7 FPS).
*   **Fix**: Implemented **Batch Inference**. GPU processes both frames in parallel. FPS recovered.

### Phase 3: The "Ghost Diver"
*   **Issue**: Model kept finding Divers in random coral reefs.
*   **Fix**: Implemented **Gatekeeper Logic**. Bumped Diver confidence requirement to 55%.

### Phase 4: The "Submarine on Diver"
*   **Issue**: Feature overlap caused flickering between "Diver" and "Submarine".
*   **Fix**: **Smart NMS**. Hard-coded logic to kill Submarine detections if a Diver is confirmed in the same box.

---

## 6. Dataset Specifications (Latest)
*   **Sources**:
    *   `mines` (Real naval mines)
    *   `diver-real` (Human divers)
    *   `drones` (ROVs/AUVs)
    *   `underwater-submarines` (Manned subs)
*   **Mappings**:
    *   `trash`/`bio-life` -> Mapped to `-1` (Ignored/Background) to reduce false positives.
*   **Final Output**: `data.yaml` configured for 4 classes.
