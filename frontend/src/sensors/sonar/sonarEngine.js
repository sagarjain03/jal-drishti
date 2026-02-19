/**
 * sonarEngine.js
 * 
 * Orchestrator hook: useSonarEngine()
 * 
 * Produces RAW sonar data only — NO fusion logic.
 * Feeds raw data to SystemStateManager via context.
 * 
 * Returns: live detections, metrics, time-series, SNR,
 *          velocity, bearing drift, persistence, environment.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { SonarDataGenerator } from './sonarDataGenerator';
import { SonarTemporalModel } from './sonarTemporalModel';
import { computeConfidence, computeVelocity, computeSNR, computeRangeCertainty, computeDopplerCertainty } from './sonarMath';
import { TICK_INTERVAL_MS } from './sonarConfig';

/**
 * Custom React hook for the sonar sensor engine.
 * 
 * @param {function|null} onUpdate - Callback to feed raw data to SystemStateManager
 * @returns {object} Live sonar state
 */
export const useSonarEngine = (onUpdate = null) => {
  const generatorRef = useRef(new SonarDataGenerator());
  const temporalRef = useRef(new SonarTemporalModel());
  const velocityHistoryRef = useRef([]);

  const [sonarState, setSonarState] = useState(() => buildInitialState(generatorRef.current));

  useEffect(() => {
    const interval = setInterval(() => {
      const gen = generatorRef.current;
      const temporal = temporalRef.current;

      // Tick simulation
      const tickData = gen.tick();

      // Compute SNR
      const snrResult = computeSNR(tickData.signalStrength, tickData.noiseLevel);

      // Compute velocity for strongest detection
      const strongest = tickData.detections.reduce((best, d) =>
        (!best || d.confidence > best.confidence) ? d : best, null);

      let velocityResult = { velocity: 0, direction: 'STATIONARY' };
      if (strongest && strongest.prevDistance !== undefined) {
        velocityResult = computeVelocity(
          strongest.prevDistance,
          strongest.distance,
          TICK_INTERVAL_MS / 1000
        );
        velocityHistoryRef.current.push(velocityResult.velocity);
        if (velocityHistoryRef.current.length > 10) velocityHistoryRef.current.shift();
      }

      // Confidence composition
      const rangeCertainty = strongest ? computeRangeCertainty(strongest.distance) : 0;
      const dopplerCertainty = computeDopplerCertainty(velocityHistoryRef.current);
      const confidenceResult = computeConfidence({
        rangeCertainty,
        signalClarity: tickData.signalStrength,
        noisePenalty: tickData.noiseLevel,
        dopplerCertainty
      });

      // Record in temporal model
      temporal.record({
        signalStrength: tickData.signalStrength,
        noiseLevel: tickData.noiseLevel,
        detections: tickData.detections,
        snr: snrResult.snr
      });

      const timeSeries = temporal.getTimeSeries();
      const bearingDrift = temporal.getBearingDrift();
      const persistence = temporal.getPersistence();

      // Build new state
      const newState = {
        // Raw detections
        detections: tickData.detections,

        // Metrics
        signalStrength: tickData.signalStrength,
        noiseLevel: tickData.noiseLevel,
        sensorHealth: tickData.environment.sensorHealth,

        // SNR
        snr: snrResult.snr,
        snrQuality: snrResult.quality,

        // Velocity / Doppler
        velocity: velocityResult.velocity,
        velocityDirection: velocityResult.direction,

        // Confidence composition
        confidence: confidenceResult.total,
        confidenceBreakdown: confidenceResult.breakdown,

        // Time-series for graphs
        timeSeries,

        // Bearing drift
        bearingDrift,

        // Persistence timers
        persistence,

        // Environment
        environment: tickData.environment,

        // Strongest detection summary
        strongestDetection: strongest ? strongest.distance : 0,
        objectStability: persistence[strongest?.label]?.confirmed ? 'CONFIRMED' : 'TRACKING'
      };

      setSonarState(newState);

      // Feed raw data to SystemStateManager
      if (onUpdate) {
        onUpdate(newState);
      }
    }, TICK_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [onUpdate]);

  return sonarState;
};

function buildInitialState(gen) {
  const state = gen.getState();
  return {
    detections: state.detections,
    signalStrength: state.signalStrength,
    noiseLevel: state.noiseLevel,
    sensorHealth: state.environment.sensorHealth,
    snr: 0,
    snrQuality: 'MODERATE',
    velocity: 0,
    velocityDirection: 'STATIONARY',
    confidence: 0,
    confidenceBreakdown: { range: 0, signal: 0, noise: 0, doppler: 0 },
    timeSeries: { signal: [], noise: [], distance: [], snr: [] },
    bearingDrift: {},
    persistence: {},
    environment: state.environment,
    strongestDetection: 0,
    objectStability: 'TRACKING'
  };
}

export default useSonarEngine;
