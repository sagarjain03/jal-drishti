## **Phase-2: Cross-Role Integration & Interdependency Plan**

---

## **1\. Purpose of This Document**

This document defines:

* How **ML, Backend, and Frontend** connect in Phase-2  
* What each role **expects and guarantees**  
* Exact **data contracts** between roles  
* **Technologies and frameworks** used  
* **Integration steps at the end of Phase-2**  
* Common **failure points and prevention strategies**

This is the document that prevents  
“It worked on my side” problems.

---

## **2\. System-Level Architecture (Phase-2)**

### **High-Level Flow**

\[ Video / Camera Input \]  
        ↓  
\[ Backend \]  
  \- Video decode  
  \- Frame scheduling  
  \- WebSocket transport  
        ↓  
\[ ML Engine \]  
  \- Frame inference  
  \- Safety logic  
        ↓  
\[ Backend \]  
  \- Result forwarding  
        ↓  
\[ Frontend \]  
  \- Visualization  
  \- Operator alerts

### **Key Design Principle**

* **ML \= Stateless intelligence**  
* **Backend \= Stateful controller**  
* **Frontend \= Reactive visual layer**

---

## **3\. Technologies & Frameworks (Locked for Phase-2)**

| Role | Language | Frameworks / Tools |
| ----- | ----- | ----- |
| ML | Python | PyTorch, OpenCV |
| Backend | Python / Node.js | FastAPI / Express, WebSocket |
| Frontend | JavaScript / TypeScript | React, Canvas API |
| Transport | — | WebSocket (low-latency) |
| Data Format | — | JSON (strict schema) |

No role should change tech stack **after integration starts**.

---

## **4\. Role-Wise Inputs & Outputs (Contracts)**

This section is **non-negotiable**.

---

### **4.1 Backend → ML Contract**

#### **Input Sent to ML**

* **Type**: Single frame  
* **Format**:  
  * RGB image  
  * Shape: `(H, W, 3)`  
  * Data type: `uint8`  
  * Range: `0–255`

#### **Backend Guarantees**

* Sends **only one frame at a time**  
* Drops frames under overload  
* No batching  
* No normalization

#### **ML Expects**

* Clean frame  
* No video files  
* No metadata confusion

---

### **4.2 ML → Backend Contract**

#### **Output Returned by ML (Strict JSON)**

{  
  "timestamp": "ISO-8601",  
  "state": "SAFE\_MODE | POTENTIAL\_ANOMALY | CONFIRMED\_THREAT",  
  "max\_confidence": 0.0,  
  "detections": \[  
    {  
      "bbox": \[x1, y1, x2, y2\],  
      "confidence": 0.63,  
      "label": "anomaly"  
    }  
  \]  
}

#### **ML Guarantees**

* Fixed schema  
* Deterministic output  
* Conservative confidence logic  
* SAFE\_MODE on uncertainty

#### **Backend Must NOT**

* Modify confidence  
* Reinterpret state  
* Rename keys

---

### **4.3 Backend → Frontend Contract**

#### **Data Sent to Frontend**

* ML JSON output (unchanged)  
* Optional system metrics:

"system": {  
  "fps": 14.5,  
  "latency\_ms": 72  
}

#### **Backend Guarantees**

* Stable WebSocket connection  
* Same schema every frame  
* Graceful disconnect handling

---

### **4.4 Frontend Expectations**

Frontend assumes:

* JSON schema never changes  
* Bounding box coordinates are valid  
* `state` is the **single source of truth**  
* Empty detections are valid

Frontend must:

* Render safely even with missing optional fields  
* Never crash on bad data

---

## **5\. Phase-2 End-to-End Integration Plan**

### **Step 1: Schema Freeze**

* All teams agree on JSON schema  
* No changes allowed after this point

---

### **Step 2: ML ↔ Backend Dry Run**

* Backend sends static test frames  
* ML returns JSON  
* Backend logs responses  
* No Frontend involved yet

**Success Criteria**

* No crashes  
* Stable latency  
* SAFE\_MODE works

---

### **Step 3: Backend ↔ Frontend Dry Run**

* Backend sends **mock JSON**  
* Frontend renders UI  
* No ML involved yet

**Success Criteria**

* UI renders all states correctly  
* No schema assumptions break

---

### **Step 4: Full System Integration**

* Video input enabled  
* Backend streams frames  
* ML processes frames  
* Frontend renders output

**Success Criteria**

* 12–15 FPS sustained  
* No memory leaks  
* No alert spamming  
* Clear operator view

---

### **Step 5: Stress & Failure Testing**

Inject:

* Blank frames  
* Noise frames  
* ML delay  
* WebSocket disconnect

**System must**

* Enter SAFE\_MODE  
* Recover automatically  
* Never crash

---

## **6\. Expected Final Outcome (Phase-2)**

At the end of Phase-2:

* ✅ Real-time video works  
* ✅ ML runs continuously  
* ✅ Backend controls flow  
* ✅ Frontend clearly communicates uncertainty  
* ✅ System is demo-ready  
* ✅ Reviewers can interact with it

---

## **7\. Common Integration Errors & Prevention**

### **Error 1: Schema Drift**

**Cause**: One team changes JSON keys  
**Prevention**: Schema freeze \+ shared document

---

### **Error 2: FPS Collapse**

**Cause**: No frame dropping  
**Prevention**: Backend scheduling logic

---

### **Error 3: UI Flickering**

**Cause**: Frontend re-interpreting ML confidence  
**Prevention**: State-driven UI only

---

### **Error 4: Silent ML Failure**

**Cause**: Wrong normalization / bad frames  
**Prevention**: ML frame validity gate \+ SAFE\_MODE

---

### **Error 5: Blame Loop**

**Cause**: Undefined responsibilities  
**Prevention**: This document 🙂

---

## **8\. Phase-2 Completion Checklist (All Roles)**

Phase-2 is **officially complete** when:

* ML engine runs frame-by-frame for long durations  
* Backend sustains real-time streaming  
* Frontend displays all states clearly  
* SAFE\_MODE triggers correctly  
* System survives bad input  
* Demo works without explanation

---

## **9\. Final Note (Important)**

Phase-2 is not about showing *accuracy*.  
Phase-2 is about proving **engineering maturity**.

If Phase-2 works:

* Phase-3 becomes optional  
* Project becomes defensible  
* Team looks professional