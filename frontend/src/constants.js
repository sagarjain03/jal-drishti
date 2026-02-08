/**
 * constants.js
 * 
 * Central configuration for Jal-Drishti Frontend.
 * DO NOT use thresholds to infer state - state comes ONLY from backend ML.
 */

// API Configuration
export const API_BASE_URL = 'http://127.0.0.1:9000';

// System states - directly from backend ML
export const SYSTEM_STATES = {
  CONFIRMED_THREAT: 'CONFIRMED_THREAT',
  POTENTIAL_ANOMALY: 'POTENTIAL_ANOMALY',
  SAFE_MODE: 'SAFE_MODE'
};

// MILESTONE-1: Fusion states from layered sensing
export const FUSION_STATES = {
  NORMAL: 'NORMAL',
  SENSOR_ALERT: 'SENSOR_ALERT',
  POTENTIAL_ANOMALY: 'POTENTIAL_ANOMALY',
  CONFIRMED_THREAT: 'CONFIRMED_THREAT',
  SENSOR_DEGRADED: 'SENSOR_DEGRADED'
};

// MILESTONE-1: Fusion state colors
export const FUSION_STATE_COLORS = {
  NORMAL: '#22C55E',           // Green - all clear
  SENSOR_ALERT: '#3B82F6',     // Blue - sonar detection
  POTENTIAL_ANOMALY: '#F97316', // Amber - multiple sensors
  CONFIRMED_THREAT: '#EF4444', // Red - full confirmation
  SENSOR_DEGRADED: '#8B5CF6'   // Purple - partial availability
};

// MILESTONE-1: Fusion state labels
export const FUSION_STATE_LABELS = {
  NORMAL: 'ALL CLEAR',
  SENSOR_ALERT: 'SENSOR ALERT',
  POTENTIAL_ANOMALY: 'POTENTIAL ANOMALY',
  CONFIRMED_THREAT: 'CONFIRMED THREAT',
  SENSOR_DEGRADED: 'SENSOR DEGRADED'
};

// MILESTONE-1: Sensor role definitions for documentation
export const SENSOR_ROLES = {
  SONAR: { name: 'Sonar', role: 'Early Suspicion', nature: 'Range & Shape-based' },
  IR: { name: 'IR/Thermal', role: 'Mid-range Confirmation', nature: 'Heat/Silhouette' },
  CAMERA: { name: 'Camera + AI', role: 'Final Confirmation', nature: 'Visual Certainty' }
};

// Source states - from SourceManager
export const SOURCE_STATES = {
  IDLE: 'IDLE',
  VIDEO_ACTIVE: 'VIDEO_ACTIVE',
  CAMERA_WAITING: 'CAMERA_WAITING',
  CAMERA_ACTIVE: 'CAMERA_ACTIVE',
  ERROR: 'ERROR'
};

// State-to-color mapping (Matte Black Theme)
export const STATE_COLORS = {
  CONFIRMED_THREAT: '#EF4444',    // Red - immediate attention
  POTENTIAL_ANOMALY: '#F97316',   // Amber - needs verification
  SAFE_MODE: '#22C55E'            // Green - normal operation
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
  STATE_CHANGE: 'state_change',
  SOURCE_SWITCH: 'source_switch'
};

// Event icons for timeline
export const EVENT_ICONS = {
  detection: '🎯',
  safe_mode_entry: '⚠️',
  safe_mode_exit: '✅',
  connection: '🔗',
  disconnection: '🔌',
  state_change: '📊',
  source_switch: '🔄'
};

// Matte Black Theme Colors
export const THEME_COLORS = {
  // Backgrounds
  bgPrimary: '#0A0A0A',      // Deep black
  bgSecondary: '#0D0D0D',    // Slightly lighter black
  bgPanel: '#121212',        // Panel background
  bgCard: '#161616',         // Card background

  // Accents - Muted
  accentGreen: '#22C55E',    // Professional green
  accentCyan: '#6B7280',     // Muted gray (was cyan)
  accentRed: '#EF4444',      // Red
  accentAmber: '#F97316',    // Warning amber

  // Borders
  borderPrimary: '#262626',  // Dark gray border
  borderGlow: '#26262640',   // Subtle border

  // Text
  textPrimary: '#E5E5E5',    // Light gray
  textSecondary: '#A3A3A3',  // Muted gray
  textMuted: '#737373'       // Very muted
};

// Reconnection config
export const RECONNECT_CONFIG = {
  MAX_ATTEMPTS: 10,
  BASE_DELAY_MS: 1000,
  MAX_DELAY_MS: 30000
};

// Overlay opacity levels
export const OVERLAY_OPACITY = {
  RECONNECTING: 0.5,
  FAILED: 0.7
};

// WebSocket config
export const WS_CONFIG = {
  URL: 'ws://127.0.0.1:9000/ws/stream',
  RAW_URL: 'ws://127.0.0.1:9000/ws/raw_feed',
  FRAME_INTERVAL_MS: 66
};

// Input source types
export const INPUT_SOURCES = {
  DUMMY_VIDEO: 'Dummy Video',
  LIVE_CAMERA: 'Live Camera',
  PHONE_CAMERA: 'Phone Camera'
};

