# Jal-Drishti Architecture Update (v3) — Complete File Breakdown

## 1. Overview

The v3 update introduces a **Centralized Engine-Driven Architecture**. This document details every file involved in the new Sensor Fusion and System State logic.

### Directory Structure

- `frontend/src/system/` — Core "Brain" (State, Confidence, Alerts)
- `frontend/src/sensors/` — "Drivers" (Raw Data Generation)
- `frontend/src/context/` — "Bridge" (React State Management)

---

## 2. System Layer (`frontend/src/system`)

_The decision-making core. Independent of UI._

### 2.1 `systemConfig.js`

- **What:** The "Constitution" of the system. Contains all tunable constants, thresholds, and magic numbers.
- **Why:** Ensures single source of truth. UI never hardcodes values like "0.75" or "3 seconds".
- **Key Configs:**
  - `CONFIDENCE_PIPELINE`: Strict steps for calculating risk.
  - `NOISE_SUPPRESSION_THRESHOLD` (0.6): Above this, correlation is disabled.
  - `PERSISTENCE_CONFIRMATION_TIME_S` (3s): Time to reach full persistence confidence.

### 2.2 `stateMachine.js`

- **What:** A class managing the transition between `SAFE_MODE`, `POTENTIAL_ANOMALY`, and `CONFIRMED_THREAT`.
- **Why:** Prevents "status flickering" where the system jumps rapidly between SAFE and THREAT.
- **Logic:**
  - **Dwell Time:** Locks state in `CONFIRMED_THREAT` for 2 seconds (`CONFIRMED_DWELL_TIME_MS`) to ensure operators see the alert.
  - **Hysteresis:** Requires 0.5s of stable high confidence to upgrade from POTENTIAL to CONFIRMED.
  - **Blocking:** High noise prevents _upward_ transitions (to avoid false alarms) but allows _downward_ ones (safety first).

### 2.3 `systemStateManager.js`

- **What:** The singleton "Controller" class. It ingests frames from WebSocket and raw data from Sonar/IR engines.
- **Why:** Centralizes the "Fusion" logic so the Dashboard, Sonar Page, and Alerts all see the exact same risk score.
- **Key Function:** `processFrame(frame)` runs the **6-Step Confidence Pipeline**:
  1. Base ML Confidence
  2. × Temporal Stability
  3. × Persistence Factor
  4. × Sensor Health Scaling
  5. - Correlation Boost (if low noise)
  6. Clamp & Round

### 2.4 `alertManager.js`

- **What:** Manages the system alert log.
- **Why:** To prevent "alert spam" when confidence wavers near a threshold.
- **Key Logic:**
  - **Debounce:** Ignores duplicate alerts within 3 seconds (`ALERT_DEBOUNCE_MS`).
  - **Badges:** Automatically adds "LOW SIGNAL RELIABILITY" or "SENSOR OFFLINE" badges to alerts based on system state.

### 2.5 `correlationEngine.js`

- **What:** The "Fusion" logic. Cross-references data from different sources.
- **Why:** To reward multi-sensor confirmation without allowing sensors to override the primary ML model.
- **Logic:**
  - Matches Sonar distance vs. Optical depth estimation.
  - Matches IR "High Heat" zones vs. Sonar detection angles.
  - Calculates a `correlationBoost` (max +0.15) added to the final confidence.

### 2.6 `temporalBuffer.js`

- **What:** Tracks visual objects over time using IOU (Intersection over Union).
- **Why:** Raw ML detections are stateless (frame-by-frame). We need to know "Is this the same boat we saw 1 second ago?"
- **Logic:**
  - Assigns unique IDs to tracked objects.
  - Calculates `persistenceFactor` (gradual ramp-up from 0.0 to 1.0 over 3 seconds).
  - Handles "decay" — objects don't vanish instantly if missed for 1 frame.

---

## 3. Sensor Engines (`frontend/src/sensors`)

_Simulation drivers that produce raw data. They do NOT calculate risk._

### 3.1 Sonar Subsystem (`sensors/sonar/`)

#### `sonarConfig.js`

- **What:** Constants for Sonar simulation (Max Range 500m, SNR thresholds, drift rates).
- **Why:** Easy tuning of simulation physics without touching code.

#### `sonarDataGenerator.js`

- **What:** The physics engine. Simulates "ticks" of data.
- **Why:** Creates realistic behavior like "turbidity reduces signal strength" and "low sensor health increases noise".
- **Logic:**
  - Updates target positions (velocity, angle).
  - Calculates `signalStrength` and `noiseLevel` based on simulated environment.

#### `sonarMath.js`

- **What:** Pure math functions.
- **Why:** Keeps the logic testable and clean.
- **Functions:** `computeSNR` (dB), `computeDoppler` (velocity from distance change), `computeConfidence` (weighted sum).

#### `sonarTemporalModel.js`

- **What:** Rolling buffers for _graphs_ (Signal, Noise, SNR history).
- **Why:** The UI needs history to draw line charts. This separates "graph history" from "logic history".

#### `sonarEngine.js`

- **What:** The React Hook (`useSonarEngine`).
- **Why:** Connects the generator to the React lifecycle. Ticks every 500ms and returns fresh state.

---

### 3.2 Infrared Subsystem (`sensors/infrared/`)

#### `infraredConfig.js`

- **What:** Constants for IR simulation (Zone definitions, material properties, drift rates).
- **Why:** Defines the valid ranges for "Organic" vs "Metallic" signatures.

#### `thermalDataGenerator.js`

- **What:** The physics engine for heat.
- **Why:** Simulates thermal inertia (objects cool down slowly) rather than random flashing numbers.
- **Logic:**
  - Generates a 12x12 Heatmap Grid.
  - Simulates sensor degradation (health drop = noisier temp readings).

#### `thermalMath.js`

- **What:** Pure math functions.
- **Why:** Calculates derived metrics like Stability Index and Material Classification.
- **Functions:** `computeStabilityIndex` (std dev), `classifyMaterial` (based on inertia/diffusion), `computeDriftRate`.

#### `infraredEngine.js`

- **What:** The React Hook (`useInfraredEngine`).
- **Why:** Connects the generator to React. Ticks every 700ms and returns fresh state.

---

## 4. Context Layer (`frontend/src/context`)

_The glue connecting Logic to UI._

### 4.1 `StreamContext.jsx`

- **What:** Manages the WebSocket connection.
- **Why:** Ensures the connection persits when navigating between pages (Home -> Sonar -> Home).
- **Logic:** Wraps `useLiveStream` and exposes the raw `frame` data.

### 4.2 `SystemStateContext.jsx`

- **What:** The "Grand Central Station" of data.
- **Why:** This is the _only_ place the UI should read data from.
- **Responsibility:**
  - Instantiates `SystemStateManager`.
  - Receives `frame` from `StreamContext`.
  - Receives `sonarData` and `irData` from the sensor hooks.
  - Passes the unified `globalState` to `App.jsx` and all children.
