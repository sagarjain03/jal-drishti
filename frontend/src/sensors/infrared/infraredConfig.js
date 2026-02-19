/**
 * infraredConfig.js
 * 
 * ALL tunable IR constants. UI files NEVER hardcode IR values.
 * Change values here → infrared pages update automatically.
 */

// ─── Zone Definitions ─────────────────────────────────────────
export const IR_ZONES = [
  { id: 'Z1', label: 'Zone Alpha', x: 3, y: 2, radius: 3, baseTemp: 22 },
  { id: 'Z2', label: 'Zone Bravo', x: 7, y: 6, radius: 2, baseTemp: 18 },
  { id: 'Z3', label: 'Zone Charlie', x: 5, y: 8, radius: 2.5, baseTemp: 15 }
];

// ─── Temperature ──────────────────────────────────────────────
export const BACKGROUND_TEMP = 12;          // °C ambient water temperature
export const ANOMALY_TEMP_THRESHOLD = 4;    // °C above background = anomaly
export const HIGH_ANOMALY_THRESHOLD = 8;    // °C above background = high anomaly

// ─── Heatmap ──────────────────────────────────────────────────
export const HEATMAP_GRID_SIZE = 12;

// ─── Material Signature Weights ───────────────────────────────
export const MATERIAL_WEIGHTS = {
  thermalInertia: 0.40,
  diffusionRate:  0.35,
  shapeCoherence: 0.25
};

export const MATERIAL_THRESHOLDS = {
  metallic: 0.7,
  organic:  0.4,
  ambient:  0.0   // below organic = ambient
};

// ─── Thermal Drift ────────────────────────────────────────────
export const DRIFT_RATE_PER_TICK = 0.3;     // °C max drift per tick
export const DRIFT_ALERT_THRESHOLD = 1.5;   // °C/s drift rate warning

// ─── Area Measurement ─────────────────────────────────────────
export const PIXEL_TO_M2_FACTOR = 0.04;     // each grid cell = 0.04 m²

// ─── Confidence Trend ─────────────────────────────────────────
export const IR_CONFIDENCE_BUFFER_SIZE = 30; // rolling buffer entries

// ─── Dynamic Degradation ─────────────────────────────────────
/**
 * When sensorHealth drops:
 * - Stability fluctuation increases
 * - Temperature readings become noisier  
 */
export const IR_HEALTH_STABILITY_AMPLIFICATION = 0.3;
export const IR_HEALTH_TEMP_NOISE = 1.5;    // °C noise amplitude at 0% health

// ─── Tick Interval ────────────────────────────────────────────
export const IR_TICK_INTERVAL_MS = 1000;

// ─── Sensor Health Default ────────────────────────────────────
export const IR_SENSOR_HEALTH_DEFAULT = 0.92;
export const IR_HEALTH_DRIFT = 0.002;       // very slow degradation
