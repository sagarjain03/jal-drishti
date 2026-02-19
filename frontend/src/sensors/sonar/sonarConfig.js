/**
 * sonarConfig.js
 * 
 * ALL tunable sonar constants. UI files NEVER hardcode sonar values.
 * Change values here → sonar pages update automatically.
 */

// ─── Range & Zones ────────────────────────────────────────────
export const MAX_RANGE = 500;         // meters
export const ZONES = [
  { range: 50,  label: 'DANGER',    color: '#EF4444', opacity: 0.1 },
  { range: 150, label: 'WARNING',   color: '#F97316', opacity: 0.08 },
  { range: 500, label: 'DETECTION', color: '#22C55E', opacity: 0.05 }
];

// ─── Confidence Composition Weights ───────────────────────────
export const CONFIDENCE_WEIGHTS = {
  range:   0.35,
  signal:  0.30,
  noise:   0.20,
  doppler: 0.15
};

// ─── SNR ──────────────────────────────────────────────────────
export const SNR_GOOD_THRESHOLD = 15;      // dB – good signal quality
export const SNR_POOR_THRESHOLD = 5;       // dB – poor signal quality

// ─── Time-Series Buffer ──────────────────────────────────────
export const TIME_SERIES_BUFFER_S = 30;    // rolling buffer in seconds
export const TICK_INTERVAL_MS = 1000;      // simulation tick interval

// ─── Persistence Timer ───────────────────────────────────────
export const PERSISTENCE_CONFIRM_FRAMES = 15; // frames required to confirm
export const PERSISTENCE_DECAY_RATE = 0.95;

// ─── Bearing Drift ───────────────────────────────────────────
export const MAX_BEARING_DRIFT_DEG_PER_S = 5;

// ─── Environmental Defaults ──────────────────────────────────
export const ENVIRONMENT_DEFAULTS = {
  turbidity: 0.3,          // [0, 1] — higher reduces signal
  salinity: 0.5,           // [0, 1] — affects propagation speed
  backgroundNoise: 0.15,   // [0, 1]
  sensorHealth: 0.95       // [0, 1] — 0 = offline
};

// ─── Environmental Drift Rates (per tick) ────────────────────
export const ENVIRONMENT_DRIFT = {
  turbidity: 0.02,
  salinity: 0.005,
  backgroundNoise: 0.03,
  sensorHealth: 0.001      // very slow degradation
};

// ─── Dynamic Degradation ─────────────────────────────────────
/**
 * When sensorHealth drops:
 * - Noise increases: effectiveNoise = baseNoise + (1 - health) * 0.4
 * - SNR decreases proportionally
 * - Signal fluctuation increases
 */
export const HEALTH_NOISE_AMPLIFICATION = 0.4;
export const HEALTH_SIGNAL_FLUCTUATION = 0.15;

// ─── Simulation Defaults ─────────────────────────────────────
export const DEFAULT_DETECTIONS = [
  { distance: 120, confidence: 0.78, angle: 45,  label: 'Object A', velocity: -2.1 },
  { distance: 340, confidence: 0.52, angle: 280, label: 'Object B', velocity: 0.5 },
  { distance: 85,  confidence: 0.91, angle: 160, label: 'Object C', velocity: -3.4 }
];

export const DEFAULT_METRICS = {
  strongestDetection: 120,
  signalStrength: 0.82,
  noiseLevel: 0.21,
  sonarConfidence: 0.78,
  objectStability: 'STABLE',
  relativeMovement: 'APPROACHING',
  snr: 18.5
};
