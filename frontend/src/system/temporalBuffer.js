/**
 * temporalBuffer.js
 * 
 * Tracks unique ML detection objects across frames via IOU overlap.
 * COMPLETELY SEPARATE from sonar temporal model.
 * 
 * Uses gradual DECAY instead of hard 5s cut:
 *   persistence_factor = min(1, tracked_duration_s / PERSISTENCE_CONFIRMATION_TIME_S)
 */

import {
  TEMPORAL_BUFFER_WINDOW_S,
  IOU_THRESHOLD,
  CENTER_DISTANCE_THRESHOLD,
  CONFIDENCE_SIMILARITY_THRESHOLD,
  PERSISTENCE_CONFIRMATION_TIME_S
} from './systemConfig';

export class TemporalBuffer {
  constructor() {
    this.trackedObjects = [];
    this._nextId = 1;
  }

  /**
   * Process a new frame of ML detections.
   * @param {Array} detections - ML detection array [{bbox, confidence, label, ...}]
   * @returns {{ similarObjectCount, trackedObjects }}
   */
  processFrame(detections = []) {
    const now = Date.now();
    const cutoffTime = now - TEMPORAL_BUFFER_WINDOW_S * 1000;

    // Remove expired objects (hard cutoff for memory, but persistence decays gradually)
    this.trackedObjects = this.trackedObjects.filter(obj => obj.lastSeenTimestamp > cutoffTime);

    // Match new detections to tracked objects
    const matched = new Set();

    for (const detection of detections) {
      let bestMatch = null;
      let bestIOU = 0;

      for (let i = 0; i < this.trackedObjects.length; i++) {
        if (matched.has(i)) continue;

        const tracked = this.trackedObjects[i];
        const iou = this._computeIOU(detection.bbox, tracked.latestBbox);
        const centerDist = this._centerDistance(detection.bbox, tracked.latestBbox);
        const confDiff = Math.abs((detection.confidence || 0) - tracked.avgConfidence);

        // Match criteria: IOU > threshold OR (close center + similar confidence)
        if (iou > IOU_THRESHOLD || 
            (centerDist < CENTER_DISTANCE_THRESHOLD && confDiff < CONFIDENCE_SIMILARITY_THRESHOLD)) {
          if (iou > bestIOU || (!bestMatch && centerDist < CENTER_DISTANCE_THRESHOLD)) {
            bestMatch = i;
            bestIOU = iou;
          }
        }
      }

      if (bestMatch !== null) {
        // Update existing tracked object
        const tracked = this.trackedObjects[bestMatch];
        tracked.lastSeenTimestamp = now;
        tracked.latestBbox = detection.bbox || tracked.latestBbox;
        tracked.bboxHistory.push(detection.bbox);
        if (tracked.bboxHistory.length > 30) tracked.bboxHistory.shift();
        tracked.avgConfidence = (tracked.avgConfidence * 0.7) + ((detection.confidence || 0) * 0.3);
        tracked.frameCount += 1;
        matched.add(bestMatch);
      } else {
        // New object
        this.trackedObjects.push({
          id: this._nextId++,
          firstSeenTimestamp: now,
          lastSeenTimestamp: now,
          latestBbox: detection.bbox || [0, 0, 0, 0],
          bboxHistory: [detection.bbox || [0, 0, 0, 0]],
          avgConfidence: detection.confidence || 0,
          label: detection.label || 'Unknown',
          frameCount: 1
        });
      }
    }

    return this.getState();
  }

  /**
   * Get current state with decay-based persistence factors.
   */
  getState() {
    const now = Date.now();

    const objectsWithPersistence = this.trackedObjects.map(obj => {
      const trackedDurationS = (now - obj.firstSeenTimestamp) / 1000;
      const persistenceFactor = Math.min(1, trackedDurationS / PERSISTENCE_CONFIRMATION_TIME_S);

      return {
        ...obj,
        persistenceFactor: Math.round(persistenceFactor * 100) / 100,
        trackedDurationS: Math.round(trackedDurationS * 10) / 10
      };
    });

    return {
      similarObjectCount: this.trackedObjects.length,
      trackedObjects: objectsWithPersistence,
      // Average persistence across all tracked objects
      avgPersistence: objectsWithPersistence.length > 0
        ? Math.round(
            objectsWithPersistence.reduce((sum, o) => sum + o.persistenceFactor, 0) /
            objectsWithPersistence.length * 100
          ) / 100
        : 0
    };
  }

  /**
   * Compute IOU (Intersection over Union) of two bboxes.
   * bbox format: [x1, y1, x2, y2] or [x, y, w, h]
   */
  _computeIOU(bboxA, bboxB) {
    if (!bboxA || !bboxB || bboxA.length < 4 || bboxB.length < 4) return 0;

    // Normalize to [x1, y1, x2, y2]
    const a = this._normalizeBbox(bboxA);
    const b = this._normalizeBbox(bboxB);

    const xOverlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const yOverlap = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const intersection = xOverlap * yOverlap;

    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
  }

  _normalizeBbox(bbox) {
    // If width/height format, convert to x1,y1,x2,y2
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
      return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]];
    }
    return bbox;
  }

  _centerDistance(bboxA, bboxB) {
    if (!bboxA || !bboxB || bboxA.length < 4 || bboxB.length < 4) return Infinity;
    const a = this._normalizeBbox(bboxA);
    const b = this._normalizeBbox(bboxB);
    const cxA = (a[0] + a[2]) / 2, cyA = (a[1] + a[3]) / 2;
    const cxB = (b[0] + b[2]) / 2, cyB = (b[1] + b[3]) / 2;
    return Math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2);
  }

  clear() {
    this.trackedObjects = [];
  }
}

export default TemporalBuffer;
