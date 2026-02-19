/**
 * systemConfig.js
 * 
 * CENTRAL SYSTEM CONFIGURATION — Jal-Drishti Frontend
 * 
 * ALL tunable constants live here. UI files NEVER hardcode values.
 * 
 * ═══════════════════════════════════════════════════════════════
 * CONFIDENCE PIPELINE (Strict Order — DO NOT reorder)
 * ═══════════════════════════════════════════════════════════════
 * 
 *   Step 1 → Base ML confidence (from WebSocket)
 *   Step 2 → × stability_index
 *   Step 3 → × persistence_factor = min(1, duration / PERSISTENCE_CONFIRMATION_TIME_S)
 *   Step 4 → × sonar_health_factor × ir_health_factor
 *   Step 5 → + correlation_boost (capped at MAX_CORRELATION_BOOST)
 *             NOTE: if noise > NOISE_SUPPRESSION_THRESHOLD → boost = 0
 *   Step 6 → clamp(MIN_CONFIDENCE_FLOOR, MAX_CONFIDENCE_CAP)
 *             → round to 2 decimal places
 *             → Final confidence drives state machine
 * 
 * ═══════════════════════════════════════════════════════════════
 */

// ─── State Thresholds ─────────────────────────────────────────
export const SAFE_THRESHOLD = 0.40;
export const CONFIRMED_THRESHOLD = 0.75;

// ─── Confidence Pipeline ──────────────────────────────────────
export const MIN_CONFIDENCE_FLOOR = 0.05;
export const MAX_CONFIDENCE_CAP = 0.98;
export const MAX_CORRELATION_BOOST = 0.15;

/**
 * Persistence Factor
 * persistence_factor = min(1, tracked_duration_s / PERSISTENCE_CONFIRMATION_TIME_S)
 * Smooth ramp-up, not binary step.
 */
export const PERSISTENCE_CONFIRMATION_TIME_S = 3;

// ─── Noise Suppression ───────────────────────────────────────
/**
 * When noise_level > NOISE_SUPPRESSION_THRESHOLD:
 *   1. Block UPWARD state transitions only (do NOT auto-downgrade)
 *   2. Set correlation boost = 0
 *   3. Add "Low Signal Reliability" badge
 *   4. Reduce confidence gradually via noise_penalty_factor:
 *
 *      noise_penalty_factor = max(0.7, 1 - (noise_level / MAX_NOISE) * 0.3)
 *
 *   This ensures penalty never drops below 70% of original value.
 *   Decay is linear over ~1 second of sustained high noise.
 */
export const NOISE_SUPPRESSION_THRESHOLD = 0.6;
export const MAX_NOISE = 1.0;
export const NOISE_PENALTY_FLOOR = 0.7;
export const NOISE_PENALTY_SCALE = 0.3;

/**
 * Compute noise penalty factor.
 * @param {number} noiseLevel - Current noise level [0, 1]
 * @returns {number} Factor ∈ [NOISE_PENALTY_FLOOR, 1.0]
 */
export const computeNoisePenaltyFactor = (noiseLevel) => {
  return Math.max(
    NOISE_PENALTY_FLOOR,
    1 - (noiseLevel / MAX_NOISE) * NOISE_PENALTY_SCALE
  );
};

// ─── State Machine ────────────────────────────────────────────
/**
 * Minimum dwell time in ms.
 * After entering CONFIRMED_THREAT, state is locked for this duration
 * before any downgrade is allowed.
 */
export const CONFIRMED_DWELL_TIME_MS = 2000;

/**
 * Hysteresis: require 0.5s of stable confidence above CONFIRMED_THRESHOLD
 * before transitioning POTENTIAL → CONFIRMED.
 */
export const CONFIRMED_HYSTERESIS_MS = 500;

// ─── Alert Manager ────────────────────────────────────────────
export const ALERT_DEBOUNCE_MS = 3000;
export const MAX_ALERT_HISTORY = 50;

// ─── Temporal Buffer (ML Detections Only) ─────────────────────
export const TEMPORAL_BUFFER_WINDOW_S = 5;
export const IOU_THRESHOLD = 0.6;
export const CENTER_DISTANCE_THRESHOLD = 50; // pixels
export const CONFIDENCE_SIMILARITY_THRESHOLD = 0.15;

// ─── Sensor Offline Handling ──────────────────────────────────
/**
 * If sonar_health === 0 OR ir_health === 0:
 *   1. Force SAFE_MODE
 *   2. Disable correlation boost
 *   3. Set safeModeReason = SENSOR_OFFLINE
 *   4. Add UI badge
 */
export const SENSOR_OFFLINE_THRESHOLD = 0;

// ─── SAFE_MODE Reason Codes ───────────────────────────────────
export const SAFE_MODE_REASONS = {
  LOW_CONFIDENCE: 'LOW_CONFIDENCE',
  HIGH_NOISE: 'HIGH_NOISE',
  SENSOR_OFFLINE: 'SENSOR_OFFLINE',
  MANUAL: 'MANUAL',
  INITIAL: 'INITIAL'
};

// ─── Fusion Stability (Optional Advanced) ─────────────────────
/**
 * Track rolling std dev of final confidence over 2 seconds.
 * If fusion_stability > VOLATILITY_THRESHOLD → "Volatile Confidence" badge.
 */
export const FUSION_STABILITY_WINDOW_S = 2;
export const VOLATILITY_THRESHOLD = 0.15;

// ─── Round helper ─────────────────────────────────────────────
/**
 * Round to 2 decimal places to prevent floating-point micro oscillations.
 */
export const roundConfidence = (value) => {
  return Math.round(value * 100) / 100;
};

/**
 * Clamp a value between floor and cap.
 */
export const clampConfidence = (value) => {
  return roundConfidence(
    Math.max(MIN_CONFIDENCE_FLOOR, Math.min(MAX_CONFIDENCE_CAP, value))
  );
};
