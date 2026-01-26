/**
 * constants.js
 * 
 * Central configuration for Jal-Drishti Frontend.
 * DO NOT use thresholds to infer state - state comes ONLY from backend ML.
 */

// System states - directly from backend ML
export const SYSTEM_STATES = {
  CONFIRMED_THREAT: 'CONFIRMED_THREAT',
  POTENTIAL_ANOMALY: 'POTENTIAL_ANOMALY',
  SAFE_MODE: 'SAFE_MODE'
};

// State-to-color mapping
export const STATE_COLORS = {
  CONFIRMED_THREAT: '#ef4444',    // Red - immediate attention
  POTENTIAL_ANOMALY: '#f97316',   // Orange - needs verification
  SAFE_MODE: '#6b7280'            // Gray - low confidence
};

// State display labels
export const STATE_LABELS = {
  CONFIRMED_THREAT: 'CONFIRMED THREAT',
  POTENTIAL_ANOMALY: 'POTENTIAL ANOMALY',
  SAFE_MODE: 'SAFE MODE'
};

// Connection states
export const CONNECTION_STATES = {
  CONNECTED: 'connected',
  CONNECTING: 'connecting',
  DISCONNECTED: 'disconnected',
  FAILED: 'failed'  // After MAX_ATTEMPTS, operator must intervene
};

// Reconnection config
// After MAX_ATTEMPTS, system enters FAILED state.
// Operator must intervene (manual refresh / backend check).
export const RECONNECT_CONFIG = {
  MAX_ATTEMPTS: 10,
  BASE_DELAY_MS: 1000,
  MAX_DELAY_MS: 30000
};

// Overlay opacity levels
export const OVERLAY_OPACITY = {
  RECONNECTING: 0.5,   // 50% - temporary
  FAILED: 0.7          // 70% - hard failure
};

// WebSocket config
export const WS_CONFIG = {
  URL: 'ws://127.0.0.1:8000/ws/stream',
  FRAME_INTERVAL_MS: 66  // ~15 FPS
};
