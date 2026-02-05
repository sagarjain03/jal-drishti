## **Runtime Issues, Fixes & Selective Delivery Architecture**

---

## **Section 1: Current Problems & Their Solutions**

### **Problem 1: Input Source Toggle Does Not Update UI Instantly**

**Observed Behavior**  
When the user switches between **VIDEO** and **CAMERA** sources, the backend switches correctly, but:

* The UI does not update immediately  
* The user must reload the page to see the change

---

### **Root Cause**

The frontend toggle is **backend-driven only**:

* UI waits for backend state change  
* No local (optimistic) UI update  
* No continuous reconciliation with backend source state

---

### **Solution: Optimistic UI \+ Source Status Polling**

**Design Principle**

* UI must respond immediately to user intent  
* Backend confirmation happens asynchronously  
* UI is later reconciled with backend truth

**Implementation**

1. On toggle click:  
   * Update UI state immediately (`CAMERA_WAITING`, `VIDEO_ACTIVE`)  
   * Do not wait for backend response

Call backend asynchronously:  
POST /api/source/select

2. 

Poll backend state:  
GET /api/source/status (every 500–1000 ms)

3.   
4. If mismatch detected:  
   * UI self-corrects without reload

**Result**

* Toggle feels instant  
* No page reload required  
* Backend latency hidden from user  
* Correct state always displayed

---

## **Problem 2: Frame Drops and Uneven Streaming**

**Observed Behavior**

* FPS fluctuates  
* Occasional frame drops  
* Enhanced feed sometimes lags behind raw feed

---

### **Root Cause**

Previously:

* Artificial cache expiry (`MAX_CACHE_AGE`)  
* Low target FPS  
* Frontend rendering not synchronized with browser refresh

---

### **Solution (Already Implemented – Do Not Reapply)**

✔ Removed `MAX_CACHE_AGE` frame skipping  
✔ Restored target FPS to 12  
✔ Frontend rendering via `requestAnimationFrame()`  
✔ Scheduler is pace-driven, not inference-driven

**Current Status**

* Backend FPS stable at 10–12  
* ML FPS stable at 12  
* Frame drops now only occur under genuine overload (expected)

---

## **Problem 3: Backend Creates Hundreds of WebSocket Connections at Startup**

**Observed Behavior**

* On backend start, 200+ WS connections are attempted  
* Frontend auto-closes after opening  
* Log spam and instability

---

### **Root Cause**

**Reconnect storm caused by incorrect WS lifecycle handling**:

* WebSocket closed when:  
  * Viewer is blocked  
  * Source is IDLE  
  * No frames available  
* Frontend reconnect logic interprets this as failure and retries rapidly

---

### **Solution: Persistent WebSocket Design**

**Correct Rule**

WebSocket must stay OPEN even if no frames are being sent.

**Backend**

* Never close WS due to:  
  * Viewer blocked  
  * Source switch  
  * IDLE / CAMERA\_WAITING  
* Simply skip sending frames

**Frontend**

* Reconnect only if:  
  * Network failure  
  * Server unreachable  
* Never reconnect because “no frames received”

**Result**

* Exactly ONE WS connection per tab  
* No reconnect storms  
* Stable backend and frontend behavior

---

## **Section 2: Selective Delivery (Multi-Viewer Streaming)**

---

## **What is Selective Delivery?**

Selective Delivery means:

**All viewers connect**, but **only selected viewers receive video frames**.

This is **not stream duplication** and **not public broadcasting**.

It is:

* Operator-controlled  
* Per-viewer delivery  
* Same frames, selectively routed

---

## **Why Selective Delivery is Needed**

Use cases:

* Operator sees live feed  
* Teammate (allowed) also sees feed  
* Other connected dashboards remain blank  
* No unnecessary bandwidth or processing

This is critical for:

* Command & control scenarios  
* Sensitive surveillance feeds  
* Team-restricted access

---

## **Architectural Flow (End-to-End)**

\[Video / Phone Camera\]  
        ↓  
\[Frame Scheduler\]  
        ↓  
\[ML Inference\]  
        ↓  
\[WebSocket Server\]  
        ↓  
\[ViewerManager\]  
   ├── Viewer A (allowed) → receives frames  
   ├── Viewer B (allowed) → receives frames  
   └── Viewer C (blocked) → receives nothing

---

## **Core Design Rules**

1. **Single Stream, Multiple Viewers**  
   * Frames are produced once  
   * No duplication or reprocessing  
2. **Viewer Identity**  
   * Each dashboard generates a unique `viewer_id` (UUID)  
   * Sent during WS connection  
3. **Selective Routing**  
   * ViewerManager decides per viewer:  
     * Send frame  
     * Skip frame  
4. **Blocked Viewer Behavior**  
   * WS stays open  
   * No frames sent  
   * UI shows “View disabled”  
5. **Source Switch Handling**  
   * All viewers reset counters  
   * No forced disconnects

---

## **Backend Components**

### **viewer\_manager.py**

Responsibilities:

* Track connected viewers  
* Maintain allow / deny list  
* Decide frame delivery per viewer

---

### **ws\_server.py**

Responsibilities:

* Require `viewer_id` on WS connect  
* Before sending frame:  
  * Check viewer permission  
  * Send or skip accordingly  
* Never close WS due to permission state

---

### **Viewer API**

* `GET /api/viewers/connected`  
* `POST /api/viewers/allow`  
* `POST /api/viewers/revoke`

Used by operator dashboard.

---

## **Frontend Components**

### **useLiveStream.js**

* Generates `viewer_id` per tab  
* Sends it during WS connection  
* Handles “no frames” gracefully

---

### **ConnectedViewers Panel**

* Shows all connected viewers  
* Operator can allow / block viewers  
* Changes take effect immediately

---

## **How a User (Operator) Uses This**

1. Operator opens dashboard  
2. Teammate opens dashboard on another laptop  
3. Both appear in **Connected Viewers** panel  
4. Operator:  
   * Allows teammate  
   * Blocks others  
5. Allowed viewers see:  
   * Raw feed  
   * Enhanced feed  
6. Blocked viewers:  
   * Dashboard stays connected  
   * No video shown

---

## **Testing Without Multiple Physical Laptops**

Recommended methods:

1. Same laptop:  
   * Chrome \+ Edge  
   * Normal \+ Incognito  
2. Each tab \= different viewer\_id  
3. Block / allow and observe behavior

---

## **Final State After All Fixes**

| Aspect | Status |
| ----- | ----- |
| Runtime source switching | Stable |
| UI responsiveness | Instant |
| Frame drops | Controlled |
| WebSocket stability | Fixed |
| Multi-viewer control | Deterministic |
| Architecture | Phase-3 compliant |

---

