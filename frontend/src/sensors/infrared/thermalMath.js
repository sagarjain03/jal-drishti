/**
 * thermalMath.js
 * 
 * Pure math functions for infrared computations.
 * No state, no side effects, no UI.
 */

import {
  MATERIAL_WEIGHTS,
  MATERIAL_THRESHOLDS,
  PIXEL_TO_M2_FACTOR,
  BACKGROUND_TEMP,
  ANOMALY_TEMP_THRESHOLD,
  HIGH_ANOMALY_THRESHOLD
} from './infraredConfig';

/**
 * Compute stability index from temperature history.
 * stability = 1 / (1 + stdDev(history))
 * 
 * @param {Array<number>} tempHistory - Array of temperature readings
 * @returns {number} Stability index [0, 1]
 */
export const computeStabilityIndex = (tempHistory) => {
  if (!tempHistory || tempHistory.length < 2) return 1.0;

  const mean = tempHistory.reduce((a, b) => a + b, 0) / tempHistory.length;
  const variance = tempHistory.reduce((sum, v) => sum + (v - mean) ** 2, 0) / tempHistory.length;
  const stdDev = Math.sqrt(variance);

  return Math.round((1 / (1 + stdDev)) * 100) / 100;
};

/**
 * Classify material from thermal properties.
 * 
 * score = (thermalInertia × weight) + (diffusionRate × weight) + (shapeCoherence × weight)
 * 
 * @param {object} params - { thermalInertia, diffusionRate, shapeCoherence } each [0, 1]
 * @returns {{ type, score, breakdown }}
 */
export const classifyMaterial = ({ thermalInertia = 0, diffusionRate = 0, shapeCoherence = 0 }) => {
  const inertiaComponent   = MATERIAL_WEIGHTS.thermalInertia * thermalInertia;
  const diffusionComponent = MATERIAL_WEIGHTS.diffusionRate  * diffusionRate;
  const shapeComponent     = MATERIAL_WEIGHTS.shapeCoherence * shapeCoherence;

  const score = Math.round((inertiaComponent + diffusionComponent + shapeComponent) * 100) / 100;

  let type;
  if (score >= MATERIAL_THRESHOLDS.metallic) type = 'METALLIC';
  else if (score >= MATERIAL_THRESHOLDS.organic) type = 'ORGANIC';
  else type = 'AMBIENT';

  return {
    type,
    score,
    breakdown: {
      thermalInertia: Math.round(inertiaComponent * 100) / 100,
      diffusionRate: Math.round(diffusionComponent * 100) / 100,
      shapeCoherence: Math.round(shapeComponent * 100) / 100
    }
  };
};

/**
 * Compute thermal drift rate from temperature history.
 * 
 * @param {Array<{time: number, value: number}>} tempHistory
 * @returns {{ driftRate, trend }}
 */
export const computeDriftRate = (tempHistory) => {
  if (!tempHistory || tempHistory.length < 2) return { driftRate: 0, trend: 'STABLE' };

  const recent = tempHistory.slice(-5);
  const first = recent[0];
  const last = recent[recent.length - 1];
  const timeDiffS = (last.time - first.time) / 1000;
  
  if (timeDiffS <= 0) return { driftRate: 0, trend: 'STABLE' };

  const driftRate = Math.round(((last.value - first.value) / timeDiffS) * 100) / 100;

  let trend;
  if (Math.abs(driftRate) < 0.2) trend = 'STABLE';
  else if (driftRate > 0) trend = 'HEATING';
  else trend = 'COOLING';

  return { driftRate, trend };
};

/**
 * Compute area in m² from zone radius.
 * 
 * @param {{ radius: number }} zone - Zone with radius in grid units
 * @returns {number} Area in m²
 */
export const computeAreaM2 = (zone) => {
  if (!zone || !zone.radius) return 0;
  const gridCells = Math.PI * zone.radius * zone.radius;
  return Math.round(gridCells * PIXEL_TO_M2_FACTOR * 100) / 100;
};

/**
 * Determine zone anomaly level based on heat delta.
 * 
 * @param {number} heatDelta - Temperature above background
 * @returns {string} Level: 'HIGH' | 'MEDIUM' | 'LOW'
 */
export const computeZoneLevel = (heatDelta) => {
  if (heatDelta >= HIGH_ANOMALY_THRESHOLD) return 'HIGH';
  if (heatDelta >= ANOMALY_TEMP_THRESHOLD) return 'MEDIUM';
  return 'LOW';
};

/**
 * Compute IR confidence from zone data.
 * Based on heat delta magnitude and stability.
 * 
 * @param {number} heatDelta
 * @param {number} stability
 * @returns {number} Confidence [0, 1]
 */
export const computeIRConfidence = (heatDelta, stability) => {
  const heatFactor = Math.min(1, heatDelta / (HIGH_ANOMALY_THRESHOLD * 1.5));
  const stabilityFactor = stability;
  const confidence = (heatFactor * 0.6) + (stabilityFactor * 0.4);
  return Math.round(Math.max(0, Math.min(1, confidence)) * 100) / 100;
};
