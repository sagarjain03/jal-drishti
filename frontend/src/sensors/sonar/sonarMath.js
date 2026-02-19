/**
 * sonarMath.js
 * 
 * Pure math functions for sonar computations.
 * No state, no side effects, no UI.
 */

import { CONFIDENCE_WEIGHTS, MAX_RANGE, SNR_GOOD_THRESHOLD, SNR_POOR_THRESHOLD } from './sonarConfig';

/**
 * Compute sonar confidence as weighted composition.
 * 
 * confidence = (range_weight × range_certainty)
 *            + (signal_weight × signal_clarity)
 *            - (noise_weight × noise_penalty)
 *            + (doppler_weight × doppler_certainty)
 * 
 * @returns {{ total, breakdown }}
 */
export const computeConfidence = ({ rangeCertainty, signalClarity, noisePenalty, dopplerCertainty }) => {
  const rangeComponent   = CONFIDENCE_WEIGHTS.range   * (rangeCertainty || 0);
  const signalComponent  = CONFIDENCE_WEIGHTS.signal  * (signalClarity || 0);
  const noiseComponent   = CONFIDENCE_WEIGHTS.noise   * (noisePenalty || 0);
  const dopplerComponent = CONFIDENCE_WEIGHTS.doppler * (dopplerCertainty || 0);

  const total = Math.max(0, Math.min(1,
    rangeComponent + signalComponent - noiseComponent + dopplerComponent
  ));

  return {
    total: Math.round(total * 100) / 100,
    breakdown: {
      range:   Math.round(rangeComponent * 100) / 100,
      signal:  Math.round(signalComponent * 100) / 100,
      noise:   Math.round(noiseComponent * 100) / 100,
      doppler: Math.round(dopplerComponent * 100) / 100
    }
  };
};

/**
 * Compute Doppler velocity (approach/recede).
 * Positive = receding, Negative = approaching.
 * 
 * @param {number} prevDist - Previous distance (m)
 * @param {number} currDist - Current distance (m)
 * @param {number} deltaTimeS - Time delta in seconds
 * @returns {{ velocity, direction }}
 */
export const computeVelocity = (prevDist, currDist, deltaTimeS) => {
  if (deltaTimeS <= 0) return { velocity: 0, direction: 'STATIONARY' };
  
  const velocity = (currDist - prevDist) / deltaTimeS; // m/s
  const roundedVelocity = Math.round(velocity * 10) / 10;

  let direction;
  if (Math.abs(roundedVelocity) < 0.3) direction = 'STATIONARY';
  else if (roundedVelocity < 0) direction = 'APPROACHING';
  else direction = 'RECEDING';

  return { velocity: roundedVelocity, direction };
};

/**
 * Compute Signal-to-Noise Ratio in dB.
 * 
 * @param {number} signalStrength - [0, 1]
 * @param {number} noiseLevel - [0, 1]
 * @returns {{ snr, quality }}
 */
export const computeSNR = (signalStrength, noiseLevel) => {
  // Prevent division by zero / log of zero
  const signal = Math.max(0.001, signalStrength);
  const noise = Math.max(0.001, noiseLevel);
  
  const snr = 10 * Math.log10(signal / noise);
  const roundedSNR = Math.round(snr * 10) / 10;

  let quality;
  if (roundedSNR >= SNR_GOOD_THRESHOLD) quality = 'GOOD';
  else if (roundedSNR >= SNR_POOR_THRESHOLD) quality = 'MODERATE';
  else quality = 'POOR';

  return { snr: roundedSNR, quality };
};

/**
 * Compute range certainty (how confident based on distance).
 * Closer objects have higher certainty.
 * 
 * @param {number} distance - Distance in meters
 * @returns {number} Certainty [0, 1]
 */
export const computeRangeCertainty = (distance) => {
  return Math.max(0, Math.min(1, 1 - (distance / MAX_RANGE)));
};

/**
 * Compute Doppler certainty based on velocity consistency.
 * Higher values for consistent velocity readings.
 * 
 * @param {Array<number>} velocityHistory - Recent velocity readings
 * @returns {number} Certainty [0, 1]
 */
export const computeDopplerCertainty = (velocityHistory = []) => {
  if (velocityHistory.length < 2) return 0.5;

  const mean = velocityHistory.reduce((a, b) => a + b, 0) / velocityHistory.length;
  const variance = velocityHistory.reduce((sum, v) => sum + (v - mean) ** 2, 0) / velocityHistory.length;
  const stdDev = Math.sqrt(variance);

  // Low variance = high certainty
  return Math.max(0, Math.min(1, 1 - stdDev / 5));
};
