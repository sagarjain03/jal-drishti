## **Phase-2 Role-Specific Execution Plan — ML Engineer**

---

## **1\. Role Definition (ML Engineer in Phase-2)**

### **Primary Responsibility**

The ML Engineer is responsible for ensuring that the **ML engine behaves correctly, efficiently, and safely when exposed to continuous, real-time frame input**.

Phase-2 is **not about improving accuracy**, but about proving that the ML pipeline:

* Works **frame-by-frame**  
* Is **stateless, stable, and fast**  
* Integrates cleanly with backend & frontend  
* Fails safely under uncertainty

---

## **2\. What the ML Engineer Owns (Clear Boundary)**

### **ML Engineer OWNS:**

* Frame-level inference logic  
* ML pipeline lifecycle (load once, reuse)  
* Input validation  
* Output format correctness  
* Confidence & safety logic  
* ML-side performance optimization  
* ML-side testing

### **ML Engineer DOES NOT own:**

* Video decoding  
* Frame scheduling  
* WebSocket transport  
* UI rendering  
* Alert visuals

Those belong to Backend & Frontend.

---

## **3\. Phase-2 ML Objectives (Explicit)**

By the end of Phase-2, the ML Engineer must deliver an ML engine that:

1. Accepts **single frames only**  
2. Processes frames **independently**  
3. Produces **deterministic JSON output**  
4. Runs continuously without memory leaks  
5. Maintains safety behavior under bad input  
6. Is backend- and frontend-agnostic

---

## **4\. ML Architecture in Phase-2**

### **Core Principle**

**ML engine is a pure frame-in → result-out system**

Input: One frame (RGB image array)  
↓  
Pre-processing  
↓  
FUnIE-GAN Enhancement  
↓  
Normalization Bridge  
↓  
YOLOv8-Nano Detection  
↓  
Confidence & Safety Logic  
↓  
Output: JSON \+ optional visual tensors

No video logic. No sockets. No UI logic.

---

## **5\. Detailed ML Execution Plan (Step-by-Step)**

---

### **STEP 1: Frame Contract Definition (Critical)**

#### **What to Do**

Define **exact input expectations** for the ML engine.

**Accepted Input:**

* Single frame  
* Shape: `(H, W, 3)`  
* Color: `RGB`  
* Type: `uint8` or `float32`  
* Range: `0–255`

**Rejected Input:**

* Empty frames  
* Grayscale frames  
* Batched frames  
* Video files

#### **Why**

This contract prevents:

* Silent crashes  
* Color mismatch bugs  
* Backend confusion

#### **Implementation**

Create a **Frame Validity Gate**:

def validate\_frame(frame):  
    if frame is None:  
        return False, "Frame is None"  
    if frame.size \== 0:  
        return False, "Empty frame"  
    if len(frame.shape) \!= 3:  
        return False, "Invalid dimensions"  
    if frame.shape\[2\] \!= 3:  
        return False, "Invalid channel count"  
    return True, None

#### **Expected Output**

* Invalid frames never reach the ML pipeline  
* ML engine never crashes on bad input

---

### **STEP 2: Model Lifecycle Management**

#### **What to Do**

* Load **FUnIE-GAN** and **YOLOv8-Nano** once  
* Move models to GPU (if available)  
* Switch to `eval()` mode  
* Use `torch.no_grad()`

#### **Why**

Reloading models per frame:

* Kills FPS  
* Causes memory leaks  
* Makes real-time impossible

#### **Implementation Rules**

* Models load in `__init__`  
* Never reload inside inference loop  
* FP16 optional but recommended

#### **Expected Output**

* Stable memory usage  
* Predictable latency

---

### **STEP 3: Frame Pre-Processing (GAN Side)**

#### **What to Do**

* Resize frame → `256×256`  
* Normalize to `[-1, 1]`  
* Ensure RGB order

Formula:

x\_gan \= (x\_raw − 127.5) / 127.5

#### **Why**

FUnIE-GAN was trained under **strict assumptions**.

Any deviation causes:

* Color artifacts  
* Hallucinations  
* Downstream detection errors

#### **Expected Output**

* Visually stable enhanced frames  
* No saturation or color distortion

---

### **STEP 4: GAN → YOLO Normalization Bridge (NON-NEGOTIABLE)**

#### **What to Do**

Convert GAN output to YOLO input:

x\_yolo \= (x\_gan\_out \+ 1\) / 2  
Resize → 640×640 (bilinear)

#### **Why**

This is the **\#1 real-world failure point** in integrated ML systems.

Without the bridge:

* YOLO silently fails  
* Black frames appear  
* Confidence becomes meaningless

#### **Expected Output**

* YOLO receives valid `[0,1]` tensor  
* Detection remains stable

---

### **STEP 5: YOLOv8-Nano Detection**

#### **What to Do**

* Run inference on enhanced frames  
* Extract:  
  * Bounding boxes  
  * Confidence scores  
  * Labels (generic: `anomaly`)

#### **Constraints**

* No weapon classification  
* No over-interpretation of labels

#### **Expected Output**

* Clean bounding boxes  
* Reasonable confidence distribution

---

### **STEP 6: Confidence & Safety Logic (ML-Side)**

#### **What to Do**

Map raw detections → system state:

| Confidence | State |
| ----- | ----- |
| \> 0.75 | CONFIRMED\_THREAT |
| 0.40 – 0.75 | POTENTIAL\_ANOMALY |
| \< 0.40 | SAFE\_MODE |

Also handle:

* No detections  
* Unstable outputs  
* Poor visibility

#### **Why**

Binary decisions are unsafe in defense systems.

#### **Expected Output**

* Conservative behavior  
* Explicit uncertainty

---

### **STEP 7: ML Output Contract (Strict JSON Schema)**

#### **What to Do**

Return output in **fixed schema**:

{  
  "timestamp": "...",  
  "state": "POTENTIAL\_ANOMALY",  
  "max\_confidence": 0.63,  
  "detections": \[  
    {  
      "bbox": \[x1, y1, x2, y2\],  
      "confidence": 0.63,  
      "label": "anomaly"  
    }  
  \]  
}

#### **Why**

Backend & Frontend depend on **schema stability**.

Changing keys \= breaking system.

#### **Expected Output**

* Backend can forward blindly  
* Frontend can render blindly

---

## **6\. ML-Side Testing Plan (Very Detailed)**

---

### **Test 1: Single Frame Sanity Test**

* Input: One valid image  
* Verify:  
  * No crash  
  * Correct output schema  
  * Reasonable confidence

---

### **Test 2: Bad Frame Injection**

* Inputs:  
  * Empty frame  
  * Noise frame  
  * Blank frame  
* Verify:  
  * SAFE\_MODE triggered  
  * No exception  
  * Clear error handling

---

### **Test 3: Frame Loop Stress Test**

* Run 1 frame repeatedly (1000+ times)  
* Verify:  
  * Memory stable  
  * No FPS drop  
  * No GPU leaks

---

### **Test 4: Video Simulation (Frame-by-Frame)**

* Extract frames from a video  
* Feed sequentially  
* Verify:  
  * No flickering logic errors  
  * Confidence behaves smoothly

---

### **Test 5: Backend Compatibility Test**

* Simulate backend sending frames  
* ML returns JSON only  
* Verify:  
  * No blocking calls  
  * Latency acceptable

---

## **7\. Expectations from Other Roles (For ML Engineer)**

### **From Backend Team**

* They must:  
  * Send **one frame at a time**  
  * Respect input format  
  * Handle dropped frames  
* They must NOT:  
  * Modify ML output schema  
  * Batch frames

### **From Frontend Team**

* They must:  
  * Trust ML output  
  * Render confidence clearly  
* They must NOT:  
  * Interpret labels as decisions  
  * Guess missing fields

---

## **8\. Final Deliverables from ML Role (Phase-2)**

ML Engineer must deliver:

* Stable ML engine (frame-based)  
* Strict input/output contracts  
* Safety-first confidence logic  
* Performance-verified pipeline  
* Clear documentation for integration

---

