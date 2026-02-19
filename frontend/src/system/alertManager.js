/**
 * alertManager.js
 * 
 * Manages alert history with debounce protection.
 * Only appends on VALID state machine transitions.
 * Adds noise badges and SAFE_MODE reason codes.
 */

import { ALERT_DEBOUNCE_MS, MAX_ALERT_HISTORY, SAFE_MODE_REASONS } from './systemConfig';

export class AlertManager {
  constructor() {
    this.alerts = [];
    this.lastAlertTime = 0;
  }

  /**
   * Attempt to add an alert on state transition.
   * Respects debounce — will not add if too soon after last alert.
   * 
   * @param {object} params
   * @param {string} params.previousState
   * @param {string} params.newState
   * @param {number} params.confidence
   * @param {boolean} params.lowSignalReliability
   * @param {string|null} params.safeModeReason
   * @returns {object|null} The new alert, or null if debounced
   */
  addTransitionAlert({ previousState, newState, confidence, lowSignalReliability = false, safeModeReason = null }) {
    const now = Date.now();

    // Debounce
    if (now - this.lastAlertTime < ALERT_DEBOUNCE_MS) {
      return null;
    }

    const timestamp = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });

    let type = 'INFO';
    let message = '';

    // Determine alert type and message
    if (newState === 'CONFIRMED_THREAT') {
      type = 'CRITICAL';
      message = 'THREAT CONFIRMED — ACTION REQUIRED';
    } else if (newState === 'POTENTIAL_ANOMALY') {
      type = 'WARNING';
      message = previousState === 'CONFIRMED_THREAT'
        ? 'THREAT DOWNGRADED — MONITORING'
        : 'POTENTIAL ANOMALY DETECTED';
    } else if (newState === 'SAFE_MODE') {
      type = 'INFO';
      const reasonLabel = this._getReasonLabel(safeModeReason);
      message = `THREAT CLEARED — ${reasonLabel}`;
    }

    // Build alert
    const alert = {
      id: now,
      type,
      message,
      timestamp,
      confidence,
      badges: []
    };

    // Noise badge
    if (lowSignalReliability) {
      alert.badges.push('LOW SIGNAL RELIABILITY');
    }

    // Safe mode reason badge
    if (safeModeReason && safeModeReason !== SAFE_MODE_REASONS.INITIAL) {
      alert.badges.push(safeModeReason.replace(/_/g, ' '));
    }

    // Prepend and cap
    this.alerts = [alert, ...this.alerts].slice(0, MAX_ALERT_HISTORY);
    this.lastAlertTime = now;

    return alert;
  }

  _getReasonLabel(reason) {
    switch (reason) {
      case SAFE_MODE_REASONS.LOW_CONFIDENCE: return 'LOW CONFIDENCE';
      case SAFE_MODE_REASONS.HIGH_NOISE: return 'HIGH NOISE';
      case SAFE_MODE_REASONS.SENSOR_OFFLINE: return 'SENSOR OFFLINE';
      case SAFE_MODE_REASONS.MANUAL: return 'MANUAL OVERRIDE';
      default: return 'NORMAL OPERATIONS';
    }
  }

  /**
   * Get current alert history.
   */
  getAlerts() {
    return this.alerts;
  }

  /**
   * Clear all alerts.
   */
  clear() {
    this.alerts = [];
  }
}

export default AlertManager;
