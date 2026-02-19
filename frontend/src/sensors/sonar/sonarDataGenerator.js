/**
 * sonarDataGenerator.js
 * 
 * Simulates sonar detections, signal/noise, and environmental conditions.
 * All defaults come from sonarConfig.js — never hardcoded in UI.
 * 
 * Dynamic degradation: low sensor health → more noise, worse SNR, signal fluctuation.
 */

import {
  DEFAULT_DETECTIONS,
  ENVIRONMENT_DEFAULTS,
  ENVIRONMENT_DRIFT,
  MAX_RANGE,
  HEALTH_NOISE_AMPLIFICATION,
  HEALTH_SIGNAL_FLUCTUATION
} from './sonarConfig';

export class SonarDataGenerator {
  constructor() {
    // Detections (with motion simulation)
    this.detections = DEFAULT_DETECTIONS.map(d => ({ ...d, prevDistance: d.distance }));

    // Signal & environmental
    this.signalStrength = 0.82;
    this.noiseLevel = ENVIRONMENT_DEFAULTS.backgroundNoise;

    // Environmental state
    this.environment = { ...ENVIRONMENT_DEFAULTS };
  }

  /**
   * Advance one tick of simulation.
   * @returns {{ detections, signalStrength, noiseLevel, environment }}
   */
  tick() {
    // ─── Drift environment ────────────────────────────────────
    this._driftEnvironment();

    // ─── Dynamic degradation from health ──────────────────────
    const health = this.environment.sensorHealth;
    const healthNoise = (1 - health) * HEALTH_NOISE_AMPLIFICATION;
    const healthFluctuation = (1 - health) * HEALTH_SIGNAL_FLUCTUATION;

    // ─── Update signal with environmental effects ─────────────
    // Turbidity reduces signal strength
    const turbidityEffect = 1 - (this.environment.turbidity * 0.3);
    this.signalStrength = Math.max(0.1, Math.min(1,
      0.82 * turbidityEffect + (Math.random() - 0.5) * 0.05 + (Math.random() - 0.5) * healthFluctuation
    ));

    // Noise affected by background noise + health degradation
    this.noiseLevel = Math.max(0.05, Math.min(0.9,
      this.environment.backgroundNoise + healthNoise + (Math.random() - 0.5) * 0.03
    ));

    // ─── Update detections ────────────────────────────────────
    this.detections = this.detections.map(det => {
      const prevDistance = det.distance;

      // Simulate movement
      let newDistance = det.distance + det.velocity + (Math.random() - 0.5) * 1.5;
      newDistance = Math.max(20, Math.min(MAX_RANGE, newDistance));

      // Slight bearing drift
      const bearingDrift = (Math.random() - 0.5) * 1.5;
      const newAngle = ((det.angle + bearingDrift) % 360 + 360) % 360;

      // Confidence affected by distance, signal, noise
      const distanceFactor = 1 - (newDistance / MAX_RANGE);
      const signalFactor = this.signalStrength;
      const noiseFactor = 1 - this.noiseLevel;
      let newConfidence = (distanceFactor * 0.4 + signalFactor * 0.35 + noiseFactor * 0.25);
      newConfidence = Math.max(0.1, Math.min(0.99, newConfidence + (Math.random() - 0.5) * 0.05));

      return {
        ...det,
        distance: Math.round(newDistance),
        prevDistance: prevDistance,
        angle: Math.round(newAngle * 10) / 10,
        confidence: Math.round(newConfidence * 100) / 100
      };
    });

    return {
      detections: this.detections,
      signalStrength: Math.round(this.signalStrength * 100) / 100,
      noiseLevel: Math.round(this.noiseLevel * 100) / 100,
      environment: { ...this.environment }
    };
  }

  /**
   * Apply minor random drift to environmental values.
   */
  _driftEnvironment() {
    const drift = (current, rate, min, max) => {
      const change = (Math.random() - 0.5) * rate * 2;
      return Math.max(min, Math.min(max, current + change));
    };

    this.environment.turbidity = drift(
      this.environment.turbidity, ENVIRONMENT_DRIFT.turbidity, 0.05, 0.9
    );
    this.environment.salinity = drift(
      this.environment.salinity, ENVIRONMENT_DRIFT.salinity, 0.1, 0.9
    );
    this.environment.backgroundNoise = drift(
      this.environment.backgroundNoise, ENVIRONMENT_DRIFT.backgroundNoise, 0.05, 0.7
    );
    this.environment.sensorHealth = drift(
      this.environment.sensorHealth, ENVIRONMENT_DRIFT.sensorHealth, 0.3, 1.0
    );

    // Round for display
    Object.keys(this.environment).forEach(key => {
      this.environment[key] = Math.round(this.environment[key] * 100) / 100;
    });
  }

  /**
   * Get current state without ticking.
   */
  getState() {
    return {
      detections: this.detections,
      signalStrength: this.signalStrength,
      noiseLevel: this.noiseLevel,
      environment: { ...this.environment }
    };
  }
}

export default SonarDataGenerator;
