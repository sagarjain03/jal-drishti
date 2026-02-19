/**
 * stateMachine.js
 * 
 * Explicit state machine for Jal-Drishti threat assessment.
 * 
 * ALLOWED TRANSITIONS ONLY:
 *   SAFE_MODE → POTENTIAL_ANOMALY
 *   POTENTIAL_ANOMALY → CONFIRMED_THREAT
 *   CONFIRMED_THREAT → POTENTIAL_ANOMALY
 *   POTENTIAL_ANOMALY → SAFE_MODE
 * 
 * DISALLOWED:
 *   SAFE_MODE → CONFIRMED_THREAT (must go through POTENTIAL first)
 * 
 * DWELL TIME:
 *   CONFIRMED_THREAT is locked for CONFIRMED_DWELL_TIME_MS before downgrade.
 * 
 * HYSTERESIS:
 *   POTENTIAL → CONFIRMED requires CONFIRMED_HYSTERESIS_MS of stable confidence.
 */

import { SYSTEM_STATES } from '../constants';
import {
  SAFE_THRESHOLD,
  CONFIRMED_THRESHOLD,
  CONFIRMED_DWELL_TIME_MS,
  CONFIRMED_HYSTERESIS_MS,
  clampConfidence,
  SAFE_MODE_REASONS
} from './systemConfig';

// Allowed transitions map
const ALLOWED_TRANSITIONS = {
  [SYSTEM_STATES.SAFE_MODE]: [SYSTEM_STATES.POTENTIAL_ANOMALY],
  [SYSTEM_STATES.POTENTIAL_ANOMALY]: [SYSTEM_STATES.CONFIRMED_THREAT, SYSTEM_STATES.SAFE_MODE],
  [SYSTEM_STATES.CONFIRMED_THREAT]: [SYSTEM_STATES.POTENTIAL_ANOMALY]
};

/**
 * StateMachine — manages threat state transitions with dwell time and hysteresis.
 */
export class StateMachine {
  constructor() {
    this.currentState = SYSTEM_STATES.SAFE_MODE;
    this.lastTransitionTime = Date.now();
    this.confirmedEntryTime = null;         // When CONFIRMED was entered
    this.hysteresisStartTime = null;        // When confidence first exceeded CONFIRMED threshold
    this.safeModeReason = SAFE_MODE_REASONS.INITIAL;
  }

  /**
   * Determine which state the confidence maps to, ignoring transition rules.
   */
  _confidenceToTargetState(confidence) {
    if (confidence >= CONFIRMED_THRESHOLD) return SYSTEM_STATES.CONFIRMED_THREAT;
    if (confidence >= SAFE_THRESHOLD) return SYSTEM_STATES.POTENTIAL_ANOMALY;
    return SYSTEM_STATES.SAFE_MODE;
  }

  /**
   * Check if a transition from current → next is allowed.
   */
  _isTransitionAllowed(from, to) {
    if (from === to) return false; // no self-transition
    const allowed = ALLOWED_TRANSITIONS[from] || [];
    return allowed.includes(to);
  }

  /**
   * Attempt a state transition based on final confidence.
   * 
   * @param {number} finalConfidence - Clamped, rounded confidence [0.05, 0.98]
   * @param {object} context - { noiseAboveThreshold, sensorOffline }
   * @returns {{ state, changed, safeModeReason }}
   */
  evaluate(finalConfidence, context = {}) {
    const { noiseAboveThreshold = false, sensorOffline = false } = context;
    const now = Date.now();
    const targetState = this._confidenceToTargetState(finalConfidence);

    // ─── FORCE SAFE_MODE on sensor offline ────────────────────
    if (sensorOffline) {
      return this._forceTransition(SYSTEM_STATES.SAFE_MODE, SAFE_MODE_REASONS.SENSOR_OFFLINE, now);
    }

    // ─── Same state — no transition needed ────────────────────
    if (targetState === this.currentState) {
      // Reset hysteresis if we're in POTENTIAL and confidence dropped below CONFIRMED
      if (this.currentState === SYSTEM_STATES.POTENTIAL_ANOMALY && finalConfidence < CONFIRMED_THRESHOLD) {
        this.hysteresisStartTime = null;
      }
      return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
    }

    // ─── UPWARD transition — blocked if noise too high ────────
    const isUpward = this._isUpward(this.currentState, targetState);
    if (isUpward && noiseAboveThreshold) {
      return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
    }

    // ─── Check allowed transitions ────────────────────────────
    if (!this._isTransitionAllowed(this.currentState, targetState)) {
      return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
    }

    // ─── DWELL TIME check for downgrade from CONFIRMED ────────
    if (this.currentState === SYSTEM_STATES.CONFIRMED_THREAT) {
      if (this.confirmedEntryTime && (now - this.confirmedEntryTime) < CONFIRMED_DWELL_TIME_MS) {
        return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
      }
    }

    // ─── HYSTERESIS for POTENTIAL → CONFIRMED ─────────────────
    if (this.currentState === SYSTEM_STATES.POTENTIAL_ANOMALY && 
        targetState === SYSTEM_STATES.CONFIRMED_THREAT) {
      if (!this.hysteresisStartTime) {
        this.hysteresisStartTime = now;
        return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
      }
      if ((now - this.hysteresisStartTime) < CONFIRMED_HYSTERESIS_MS) {
        return { state: this.currentState, changed: false, safeModeReason: this.safeModeReason };
      }
      // Hysteresis passed — allow transition
    }

    // ─── Execute transition ───────────────────────────────────
    return this._executeTransition(targetState, finalConfidence, now);
  }

  _isUpward(from, to) {
    const order = {
      [SYSTEM_STATES.SAFE_MODE]: 0,
      [SYSTEM_STATES.POTENTIAL_ANOMALY]: 1,
      [SYSTEM_STATES.CONFIRMED_THREAT]: 2
    };
    return (order[to] || 0) > (order[from] || 0);
  }

  _forceTransition(state, reason, now) {
    const changed = this.currentState !== state;
    this.currentState = state;
    this.safeModeReason = reason;
    this.lastTransitionTime = now;
    this.confirmedEntryTime = null;
    this.hysteresisStartTime = null;
    return { state, changed, safeModeReason: reason };
  }

  _executeTransition(targetState, confidence, now) {
    const previousState = this.currentState;
    this.currentState = targetState;
    this.lastTransitionTime = now;
    this.hysteresisStartTime = null;

    // Track CONFIRMED entry time for dwell
    if (targetState === SYSTEM_STATES.CONFIRMED_THREAT) {
      this.confirmedEntryTime = now;
    } else {
      this.confirmedEntryTime = null;
    }

    // Set SAFE_MODE reason
    if (targetState === SYSTEM_STATES.SAFE_MODE) {
      this.safeModeReason = SAFE_MODE_REASONS.LOW_CONFIDENCE;
    } else {
      this.safeModeReason = null;
    }

    return {
      state: targetState,
      changed: true,
      previousState,
      safeModeReason: this.safeModeReason
    };
  }

  /**
   * Get current state info.
   */
  getState() {
    return {
      state: this.currentState,
      safeModeReason: this.safeModeReason,
      lastTransitionTime: this.lastTransitionTime,
      confirmedEntryTime: this.confirmedEntryTime
    };
  }
}

export default StateMachine;
