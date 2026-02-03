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

// State-to-color mapping (Defence Theme)
export const STATE_COLORS = {
  CONFIRMED_THREAT: '#ff3366',    // Neon Red - immediate attention
  POTENTIAL_ANOMALY: '#ffaa00',   // Amber - needs verification
  SAFE_MODE: '#00ff88'            // Neon Green - normal operation
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

// System status - from backend system messages
export const SYSTEM_STATUS = {
  SAFE_MODE: 'safe_mode',
  RECOVERED: 'recovered',
  CONNECTED: 'connected'
};

// Event types for timeline
export const EVENT_TYPES = {
  DETECTION: 'detection',
  SAFE_MODE_ENTRY: 'safe_mode_entry',
  SAFE_MODE_EXIT: 'safe_mode_exit',
  CONNECTION: 'connection',
  DISCONNECTION: 'disconnection',
  STATE_CHANGE: 'state_change'
};

// Event icons for timeline
export const EVENT_ICONS = {
  detection: '🎯',
  safe_mode_entry: '⚠️',
  safe_mode_exit: '✅',
  connection: '🔗',
  disconnection: '🔌',
  state_change: '📊'
};

// Defence Theme Colors
export const THEME_COLORS = {
  // Backgrounds
  bgPrimary: '#0a1628',      // Deep navy
  bgSecondary: '#0d1b2a',    // Slightly lighter navy
  bgPanel: '#1b2838',        // Panel background
  bgCard: '#162231',         // Card background

  // Accents
  accentGreen: '#00ff88',    // Neon green
  accentCyan: '#00d4ff',     // Neon cyan
  accentRed: '#ff3366',      // Neon red
  accentAmber: '#ffaa00',    // Warning amber

  // Borders
  borderPrimary: '#1e3a5f',  // Subtle blue border
  borderGlow: '#00ff8840',   // Glow border

  // Text
  textPrimary: '#e0f2fe',    // Light blue-white
  textSecondary: '#64748b',  // Muted gray
  textMuted: '#475569'       // Very muted
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
  URL: 'ws://127.0.0.1:9000/ws/stream',
  FRAME_INTERVAL_MS: 66  // ~15 FPS
};

// Input source types
export const INPUT_SOURCES = {
  DUMMY_VIDEO: 'Dummy Video',
  LIVE_CAMERA: 'Live Camera',
  PHONE_CAMERA: 'Phone Camera'
};
