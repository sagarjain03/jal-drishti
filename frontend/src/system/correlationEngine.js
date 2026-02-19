/**
 * correlationEngine.js
 * 
 * SINGLE AUTHORITY for all cross-sensor correlation (Sonar ↔ IR ↔ Camera).
 * Sensors produce raw metrics only. This module handles fusion.
 * 
 * Rules:
 * - Correlation enhances — NEVER overrides ML confidence
 * - Bounded: max boost = MAX_CORRELATION_BOOST (0.15)
 * - If noise > threshold → correlation boost = 0
 * - Sensor health impacts global confidence (Step 4 of pipeline)
 */

import {
  MAX_CORRELATION_BOOST,
  NOISE_SUPPRESSION_THRESHOLD,
  SENSOR_OFFLINE_THRESHOLD
} from './systemConfig';

export class CorrelationEngine {
  constructor() {
    this._lastResult = this._emptyResult();
  }

  /**
   * Compute cross-sensor correlation.
   * 
   * @param {object} sonarData - Raw sonar engine output
   * @param {object} irData    - Raw IR engine output 
   * @param {Array}  mlDetections - ML detections from WebSocket
   * @param {number} noiseLevel - Current noise level [0, 1]
   * @returns {object} Correlation result
   */
  compute(sonarData, irData, mlDetections, noiseLevel = 0) {
    const result = this._emptyResult();

    // ─── Sensor health factors ────────────────────────────────
    result.sonarHealth = sonarData?.sensorHealth ?? 1.0;
    result.irHealth = irData?.sensorHealth ?? 1.0;
    result.sonarOnline = result.sonarHealth > SENSOR_OFFLINE_THRESHOLD;
    result.irOnline = result.irHealth > SENSOR_OFFLINE_THRESHOLD;
    result.anySensorOffline = !result.sonarOnline || !result.irOnline;

    // Health scaling factor (for pipeline Step 4)
    result.healthScalingFactor = result.sonarHealth * result.irHealth;

    // ─── Skip correlation if noise too high ───────────────────
    if (noiseLevel > NOISE_SUPPRESSION_THRESHOLD) {
      result.correlationBoost = 0;
      result.reason = 'Noise above threshold — correlation disabled';
      this._lastResult = result;
      return result;
    }

    // ─── Skip if a sensor is offline ──────────────────────────
    if (result.anySensorOffline) {
      result.correlationBoost = 0;
      result.reason = 'Sensor offline — correlation disabled';
      this._lastResult = result;
      return result;
    }

    // ─── Sonar ↔ Camera distance match ────────────────────────
    let sonarCameraScore = 0;
    if (sonarData?.detections?.length > 0 && mlDetections?.length > 0) {
      const sonarDist = sonarData.detections[0]?.distance || 0;
      // Estimate camera depth from bbox size (rough: bigger box = closer)
      const cameraEstimatedDepth = this._estimateCameraDepth(mlDetections[0]);
      const distanceDiff = Math.abs(sonarDist - cameraEstimatedDepth);
      sonarCameraScore = distanceDiff < 100 ? (1 - distanceDiff / 100) * 0.5 : 0;
    }

    // ─── IR ↔ Sonar alignment ─────────────────────────────────
    let irSonarScore = 0;
    if (irData?.zones?.length > 0 && sonarData?.detections?.length > 0) {
      // If IR detects anomaly at similar bearing as sonar
      const irHasAnomaly = irData.zones.some(z => z.level === 'HIGH' || z.level === 'MEDIUM');
      if (irHasAnomaly) {
        irSonarScore = 0.3;
      }
    }

    // ─── IR ↔ Camera alignment ────────────────────────────────
    let irCameraScore = 0;
    if (irData?.zones?.length > 0 && mlDetections?.length > 0) {
      const irHasAnomaly = irData.zones.some(z => z.level === 'HIGH');
      if (irHasAnomaly && mlDetections.length > 0) {
        irCameraScore = 0.2;
      }
    }

    // ─── Compute total correlation ────────────────────────────
    const rawBoost = sonarCameraScore + irSonarScore + irCameraScore;
    result.correlationBoost = Math.min(rawBoost, MAX_CORRELATION_BOOST);
    result.correlationScore = Math.min(1, sonarCameraScore + irSonarScore + irCameraScore);

    // Breakdown
    result.breakdown = {
      sonarCamera: Math.round(sonarCameraScore * 100) / 100,
      irSonar: Math.round(irSonarScore * 100) / 100,
      irCamera: Math.round(irCameraScore * 100) / 100
    };

    result.reason = result.correlationScore > 0
      ? 'Multi-sensor alignment detected'
      : 'No cross-sensor correlation';

    this._lastResult = result;
    return result;
  }

  _estimateCameraDepth(detection) {
    // Rough depth estimate from bbox height (larger = closer)
    if (!detection) return 500;
    const bboxHeight = detection.bbox?.[3] || detection.height || 50;
    // Inverse relationship: bigger box = closer
    return Math.max(20, 500 - (bboxHeight * 5));
  }

  _emptyResult() {
    return {
      correlationScore: 0,
      correlationBoost: 0,
      healthScalingFactor: 1.0,
      sonarHealth: 1.0,
      irHealth: 1.0,
      sonarOnline: true,
      irOnline: true,
      anySensorOffline: false,
      breakdown: { sonarCamera: 0, irSonar: 0, irCamera: 0 },
      reason: 'No data'
    };
  }

  getLastResult() {
    return this._lastResult;
  }
}

export default CorrelationEngine;
