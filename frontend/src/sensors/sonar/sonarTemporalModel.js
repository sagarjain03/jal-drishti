/**
 * sonarTemporalModel.js
 * 
 * Rolling time-series buffers for SIMULATED SONAR SIGNALS.
 * NOT for ML bounding boxes — those go to temporalBuffer.js.
 * 
 * Tracks:
 * - Signal strength history
 * - Noise level history
 * - Distance history
 * - SNR history
 * - Bearing drift history
 * - Persistence timers per detection
 */

import { TIME_SERIES_BUFFER_S, PERSISTENCE_CONFIRM_FRAMES, PERSISTENCE_DECAY_RATE } from './sonarConfig';

export class SonarTemporalModel {
  constructor() {
    this.bufferSize = TIME_SERIES_BUFFER_S;

    // Rolling buffers
    this.signalHistory = [];
    this.noiseHistory = [];
    this.distanceHistory = [];
    this.snrHistory = [];
    this.bearingHistory = [];

    // Persistence tracking per detection label
    this._persistenceCounters = {};
  }

  /**
   * Record a tick of sonar data.
   */
  record({ signalStrength, noiseLevel, detections = [], snr = 0 }) {
    const now = Date.now();
    const timestamp = now;

    // Push to rolling buffers
    this.signalHistory.push({ time: timestamp, value: signalStrength });
    this.noiseHistory.push({ time: timestamp, value: noiseLevel });
    this.snrHistory.push({ time: timestamp, value: snr });

    // Distance: track strongest detection
    const strongestDet = detections.reduce((best, d) =>
      (!best || d.confidence > best.confidence) ? d : best, null);
    if (strongestDet) {
      this.distanceHistory.push({ time: timestamp, value: strongestDet.distance });
    }

    // Bearing: track all detection angles
    for (const det of detections) {
      this.bearingHistory.push({ time: timestamp, label: det.label, bearing: det.angle });
    }

    // Trim buffers
    this._trim();

    // Update persistence
    this._updatePersistence(detections);
  }

  /**
   * Trim buffers to rolling window.
   */
  _trim() {
    const cutoff = Date.now() - this.bufferSize * 1000;
    this.signalHistory = this.signalHistory.filter(e => e.time > cutoff);
    this.noiseHistory = this.noiseHistory.filter(e => e.time > cutoff);
    this.distanceHistory = this.distanceHistory.filter(e => e.time > cutoff);
    this.snrHistory = this.snrHistory.filter(e => e.time > cutoff);
    this.bearingHistory = this.bearingHistory.filter(e => e.time > cutoff);
  }

  /**
   * Update persistence timers for detections.
   * A detection is "confirmed" when its counter reaches PERSISTENCE_CONFIRM_FRAMES.
   */
  _updatePersistence(detections) {
    const activeLabels = new Set(detections.map(d => d.label));

    // Increment seen detections
    for (const label of activeLabels) {
      if (!this._persistenceCounters[label]) {
        this._persistenceCounters[label] = { count: 0, confirmed: false };
      }
      this._persistenceCounters[label].count += 1;
      if (this._persistenceCounters[label].count >= PERSISTENCE_CONFIRM_FRAMES) {
        this._persistenceCounters[label].confirmed = true;
      }
    }

    // Decay unseen detections
    for (const label of Object.keys(this._persistenceCounters)) {
      if (!activeLabels.has(label)) {
        this._persistenceCounters[label].count = Math.floor(
          this._persistenceCounters[label].count * PERSISTENCE_DECAY_RATE
        );
        if (this._persistenceCounters[label].count <= 0) {
          delete this._persistenceCounters[label];
        }
      }
    }
  }

  /**
   * Get time-series data for graphs.
   */
  getTimeSeries() {
    return {
      signal: this.signalHistory.map(h => ({ time: h.time, value: h.value })),
      noise: this.noiseHistory.map(h => ({ time: h.time, value: h.value })),
      distance: this.distanceHistory.map(h => ({ time: h.time, value: h.value })),
      snr: this.snrHistory.map(h => ({ time: h.time, value: h.value }))
    };
  }

  /**
   * Get bearing drift for each tracked object.
   */
  getBearingDrift() {
    const driftByLabel = {};
    for (const entry of this.bearingHistory) {
      if (!driftByLabel[entry.label]) driftByLabel[entry.label] = [];
      driftByLabel[entry.label].push(entry.bearing);
    }

    const result = {};
    for (const [label, bearings] of Object.entries(driftByLabel)) {
      if (bearings.length < 2) {
        result[label] = { drift: 0, trend: 'STABLE' };
        continue;
      }
      const first = bearings[0];
      const last = bearings[bearings.length - 1];
      const totalDrift = last - first;
      result[label] = {
        drift: Math.round(totalDrift * 10) / 10,
        trend: Math.abs(totalDrift) < 2 ? 'STABLE' : totalDrift > 0 ? 'CLOCKWISE' : 'COUNTER-CW'
      };
    }
    return result;
  }

  /**
   * Get persistence state for each tracked detection.
   */
  getPersistence() {
    const result = {};
    for (const [label, data] of Object.entries(this._persistenceCounters)) {
      result[label] = {
        frames: data.count,
        confirmed: data.confirmed,
        progress: Math.min(1, data.count / PERSISTENCE_CONFIRM_FRAMES)
      };
    }
    return result;
  }
}

export default SonarTemporalModel;
