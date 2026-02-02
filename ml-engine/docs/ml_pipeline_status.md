# ML Pipeline Status & Technical Roadmap
**Date:** 2026-02-01
**Version:** 2.1 (Optimized)

## 1. Current System Status
The pipeline has been optimized for a balance of **Performance (FPS)**, **Clarity**, and **Stability**.

### A. Pipeline Configuration
| Component | Setting | Reason |
| :--- | :--- | :--- |
| **Resolution** | **256x256** (Native) | Ensures High FPS (6-10). Higher resolutions caused significant lag. |
| **Resizing** | **PyTorch Bilinear** | `F.interpolate` provides much smoother upscaling than `cv2.resize`, preventing "blockiness" even at 256p. |
| **Sharpening** | **Disabled** | CPU-based Unsharp Mask was consuming ~20% of frame budget. Removed to prioritize FPS. |
| **Color Tone** | **Red/Warm** | The "Reddish" output is the **scientifically correct** restoration of red light absorbed by water. |
| **Visual Fixes** | **Active** | `Object-Fit: Contain` (Zoom Fix), `Clamping` (Noise Fix), `Channel Swap Restoration` (Tint Fix). |

### B. Current Limitations (To Be Addressed)
1.  **Aesthetic Tone:** The output is "Reddish/Warm" (Scientific). The user desires a "Cinematic Blue" look.
2.  **Object Detection:** YOLOv8 is using the **COCO Dataset** weights. It detects "person", "bird", "dog" because it has never seen a "diver" or "mine". It forces these matches incorrectly.

---

## 2. Technical Roadmap: Fine-Tuning Procedures

To address the limitations above, we must move from "Inference-Only" to "Training/Fine-Tuning".

### A. Fine-Tuning FUnIE-GAN (Target: "Cinematic Blue" Tone)
We must retrain the GAN to map "Murky Green" inputs to "Clear Blue" outputs instead of "Clear Red/Warm" outputs.

#### **Requirements**
1.  **Dataset (Paired)**:
    -   **Input A (Murky)**: Raw underwater images (Green/Dark).
    -   **Ground Truth B (Blue)**: The same images, but color-graded to the desired "Bluish" tone (e.g., using Photoshop or Lightroom).
    -   **Size**: ~1000-2000 paired images.
2.  **Hardware**: NVIDIA GPU (Linear T4 or better) with >12GB VRAM.

#### **Training Loop Logic**
The GAN optimizes a composite loss function:
$$L_{total} = \lambda_{GAN} L_{cGAN}(G, D) + \lambda_{L1} L_{L1}(G)$$
-   **$L_{cGAN}$**: Forces the generator to create "realistic" looking water.
-   **$L_{L1}$**: Forces the generator to match the **Blue Ground Truth** pixels exactly.

#### **Procedure**
1.  Prepare dataset in `A/` (Murky) and `B/` (Blue) folders.
2.  Load Pre-trained FUnIE-GAN weights (Transfer Learning).
3.  Run training for ~50-100 Epochs.
4.  Monitor `val_loss`. When minimal, save `G_blue_tint.pth`.
5.  Replace `funie_gan.pth` in the pipeline with this new file.

---

### B. Fine-Tuning YOLOv8 (Target: Underwater Classes)
We must teach YOLO to recognize specific underwater objects (Diver, Submarine, Mine, ROV) instead of generic COCO objects.

#### **Requirements**
1.  **Dataset (Annotated)**:
    -   Images of Divers, Mines, etc.
    -   Annotations in **YOLO Format** (`.txt` files with `class_id x_center y_center width height`).
    -   Tools: Roboflow, CVAT, or LabelImg.
2.  **Classes**: Define specific ID map: `{0: 'diver', 1: 'bio_life', 2: 'mine', 3: 'submarine'}`.

#### **Training Command (Ultralytics)**
```python
from ultralytics import YOLO

# 1. Load Base Model (Nano is fastest)
model = YOLO('yolov8n.pt') 

# 2. Train on Custom Underwater Dataset
results = model.train(
    data='underwater_config.yaml', # Points to your images/labels
    epochs=100,                    # Training duration
    imgsz=640,                     # Input size
    batch=16,                      # GPU Batch size
    device=0                       # GPU ID
)
```

#### **Integration**
1.  Training outputs `best.pt` in `runs/detect/train/weights/`.
2.  Update system config: `YOLO_WEIGHTS = "path/to/best.pt"`.
3.  The pipeline will automatically pick up the new labels (e.g., "mine" instead of "suitcase").

---

## 3. Summary
- **Current Pipeline**: Optimized for Speed and Scientific Accuracy.
- **Next Phase**: Data Collection & Training to achieve **Cinematic Aesthetics** and **Semantic Accuracy**.
