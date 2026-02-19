/**
 * systemStateManager.js
 * 
 * GLOBAL SYSTEM STATE AUTHORITY — Jal-Drishti Frontend
 * 
 * All UI components derive state from this module.
 * No component computes risk independently.
 * 
 * Implements the strict 6-step confidence pipeline:
 *   Step 1 → Base ML confidence
 *   Step 2 → × stability_index
 *   Step 3 → × persistence_factor
 *   Step 4 → × sensor health scaling
 *   Step 5 → + correlation boost (0 if noise high)
 *   Step 6 → clamp + round to 2 decimals
 */

import { StateMachine } from './stateMachine';
import { AlertManager } from './alertManager';
import { CorrelationEngine } from './correlationEngine';
import { TemporalBuffer } from './temporalBuffer';
import {
  clampConfidence,
  roundConfidence,
  computeNoisePenaltyFactor,
  NOISE_SUPPRESSION_THRESHOLD,
  SAFE_MODE_REASONS,
  FUSION_STABILITY_WINDOW_S,
  VOLATILITY_THRESHOLD
} from './systemConfig';

export class SystemStateManager {
  constructor() {
    this.stateMachine = new StateMachine();
    this.alertManager = new AlertManager();
    this.correlationEngine = new CorrelationEngine();
    this.temporalBuffer = new TemporalBuffer();

    // Pipeline breakdown for Fusion Transparency Panel
    this.pipelineBreakdown = this._emptyBreakdown();

    // Fusion stability tracking (rolling std dev of final confidence)
    this._confidenceHistory = [];
    this.fusionStability = 0;
    this.isVolatile = false;

    // Sensor raw data
    this._sonarData = null;
    this._irData = null;
    this._noiseLevel = 0;

    // State
    this.globalState = this._initialState();
  }

  /**
   * Receive raw sonar data from useSonarEngine().
   */
  updateSonarData(sonarData) {
    this._sonarData = sonarData;
    this._noiseLevel = sonarData?.noiseLevel ?? this._noiseLevel;
  }

  /**
   * Receive raw IR data from useInfraredEngine().
   */
  updateIRData(irData) {
    this._irData = irData;
  }

  /**
   * Process an ML WebSocket frame and run the full confidence pipeline.
   * 
   * @param {object} frame - Normalized frame from useLiveStream
   * @returns {object} Updated global state
   */
  processFrame(frame) {
    if (!frame) return this.globalState;

    // ─── Process ML detections through temporal buffer ────────
    const temporalState = this.temporalBuffer.processFrame(frame.detections || []);

    // ─── STEP 1: Base ML confidence ──────────────────────────
    const step1_base = frame.max_confidence ?? 0;

    // ─── STEP 2: × stability_index ───────────────────────────
    // Stability from temporal buffer persistence
    const stabilityIndex = temporalState.avgPersistence || 1.0;
    const step2_afterStability = step1_base * stabilityIndex;

    // ─── STEP 3: × persistence_factor ────────────────────────
    // Use highest individual persistence from tracked objects
    const maxPersistence = temporalState.trackedObjects.length > 0
      ? Math.max(...temporalState.trackedObjects.map(o => o.persistenceFactor))
      : 1.0;
    const step3_afterPersistence = step2_afterStability * maxPersistence;

    // ─── CORRELATION (needs to run before Step 4) ─────────────
    const correlationResult = this.correlationEngine.compute(
      this._sonarData,
      this._irData,
      frame.detections || [],
      this._noiseLevel
    );

    // ─── STEP 4: × sensor health scaling ─────────────────────
    const healthFactor = correlationResult.healthScalingFactor;
    const step4_afterHealth = step3_afterPersistence * healthFactor;

    // ─── STEP 5: + correlation boost ─────────────────────────
    // NOTE: if noise > threshold → boost already set to 0 by correlationEngine
    const step5_afterCorrelation = step4_afterHealth + correlationResult.correlationBoost;

    // ─── Noise penalty (applied if noise high) ────────────────
    const noiseAboveThreshold = this._noiseLevel > NOISE_SUPPRESSION_THRESHOLD;
    const noisePenalty = noiseAboveThreshold
      ? computeNoisePenaltyFactor(this._noiseLevel)
      : 1.0;
    const step5_withNoise = step5_afterCorrelation * noisePenalty;

    // ─── STEP 6: Clamp + round ────────────────────────────────
    const finalConfidence = clampConfidence(step5_withNoise);

    // ─── Store pipeline breakdown for Fusion Transparency ─────
    this.pipelineBreakdown = {
      step1_mlBase: roundConfidence(step1_base),
      step2_stability: roundConfidence(stabilityIndex),
      step3_persistence: roundConfidence(maxPersistence),
      step4_healthScaling: roundConfidence(healthFactor),
      step5_correlationBoost: roundConfidence(correlationResult.correlationBoost),
      step5_noisePenalty: roundConfidence(noisePenalty),
      step6_final: finalConfidence,
      noiseAboveThreshold,
      lowSignalReliability: noiseAboveThreshold
    };

    // ─── State machine evaluation ─────────────────────────────
    const machineResult = this.stateMachine.evaluate(finalConfidence, {
      noiseAboveThreshold,
      sensorOffline: correlationResult.anySensorOffline
    });

    // ─── Alert on transition ──────────────────────────────────
    if (machineResult.changed) {
      // Determine safe mode reason
      let safeModeReason = machineResult.safeModeReason;
      if (machineResult.state === 'SAFE_MODE' && noiseAboveThreshold) {
        safeModeReason = SAFE_MODE_REASONS.HIGH_NOISE;
      }

      this.alertManager.addTransitionAlert({
        previousState: machineResult.previousState,
        newState: machineResult.state,
        confidence: finalConfidence,
        lowSignalReliability: noiseAboveThreshold,
        safeModeReason
      });
    }

    // ─── Fusion stability tracking ────────────────────────────
    this._updateFusionStability(finalConfidence);

    // ─── Build global state ───────────────────────────────────
    this.globalState = {
      // Core state
      state: machineResult.state,
      confidence: finalConfidence,
      safeModeReason: machineResult.safeModeReason,

      // Raw frame data (passthrough)
      detections: frame.detections || [],
      imageData: frame.image_data,
      riskScore: frame.risk_score ?? 0,
      frameId: frame.frame_id,
      timestamp: frame.timestamp,
      system: frame.system,

      // Pipeline transparency
      pipelineBreakdown: { ...this.pipelineBreakdown },

      // Temporal buffer
      temporalBuffer: temporalState,
      similarObjectCount: temporalState.similarObjectCount,

      // Correlation
      correlation: correlationResult,

      // Sensor data
      sonarData: this._sonarData,
      irData: this._irData,
      noiseLevel: this._noiseLevel,

      // Alerts
      alerts: this.alertManager.getAlerts(),

      // Flags
      lowSignalReliability: noiseAboveThreshold,
      anySensorOffline: correlationResult.anySensorOffline,
      fusionStability: this.fusionStability,
      isVolatile: this.isVolatile,

      // Fusion data from frame (passthrough)
      sensors: frame.sensors,
      fusionState: frame.fusion_state,
      fusionMessage: frame.fusion_message,
      threatPriority: frame.threat_priority,
      signature: frame.signature,
      explainability: frame.explainability,
      seenBefore: frame.seen_before,
      occurrenceCount: frame.occurrence_count
    };

    return this.globalState;
  }

  /**
   * Track rolling std dev of final confidence for volatility detection.
   */
  _updateFusionStability(confidence) {
    const now = Date.now();
    this._confidenceHistory.push({ value: confidence, time: now });

    // Keep only last FUSION_STABILITY_WINDOW_S seconds
    const cutoff = now - FUSION_STABILITY_WINDOW_S * 1000;
    this._confidenceHistory = this._confidenceHistory.filter(h => h.time > cutoff);

    if (this._confidenceHistory.length < 3) {
      this.fusionStability = 0;
      this.isVolatile = false;
      return;
    }

    const values = this._confidenceHistory.map(h => h.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
    this.fusionStability = roundConfidence(Math.sqrt(variance));
    this.isVolatile = this.fusionStability > VOLATILITY_THRESHOLD;
  }

  _emptyBreakdown() {
    return {
      step1_mlBase: 0,
      step2_stability: 1,
      step3_persistence: 1,
      step4_healthScaling: 1,
      step5_correlationBoost: 0,
      step5_noisePenalty: 1,
      step6_final: 0.05,
      noiseAboveThreshold: false,
      lowSignalReliability: false
    };
  }

  _initialState() {
    return {
      state: 'SAFE_MODE',
      confidence: 0.05,
      safeModeReason: SAFE_MODE_REASONS.INITIAL,
      detections: [],
      imageData: null,
      riskScore: 0,
      pipelineBreakdown: this._emptyBreakdown(),
      temporalBuffer: { similarObjectCount: 0, trackedObjects: [], avgPersistence: 0 },
      similarObjectCount: 0,
      correlation: { correlationScore: 0, correlationBoost: 0, healthScalingFactor: 1, breakdown: { sonarCamera: 0, irSonar: 0, irCamera: 0 } },
      sonarData: null,
      irData: null,
      noiseLevel: 0,
      alerts: [],
      lowSignalReliability: false,
      anySensorOffline: false,
      fusionStability: 0,
      isVolatile: false,
      sensors: {},
      fusionState: 'NORMAL',
      fusionMessage: '',
      threatPriority: 'LOW',
      signature: '',
      explainability: [],
      seenBefore: false,
      occurrenceCount: 0
    };
  }

  getState() {
    return this.globalState;
  }
}

export default SystemStateManager;
