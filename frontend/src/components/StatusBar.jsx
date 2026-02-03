import React from 'react';
import { STATE_COLORS, STATE_LABELS, CONNECTION_STATES, INPUT_SOURCES } from '../constants';
import '../App.css';

/**
 * StatusBar Component (Enhanced)
 * 
 * Displays:
 * - System state badge (color-coded with defence theme)
 * - Max confidence (from backend)
 * - Latency (backend-reported only, show -- if missing)
 * - Render FPS (frontend measured)
 * - ML FPS (backend-reported, if available)
 * - Input source indicator
 * - Connection status with glow effects
 * 
 * Color Rules:
 * - Green → Normal/Connected
 * - Yellow → Degraded/Potential Anomaly
 * - Red → SAFE MODE/Threat
 */
const StatusBar = ({
    systemState = 'SAFE_MODE',
    maxConfidence = 0,
    latencyMs = null,
    renderFps = 0,
    mlFps = null,
    connectionStatus = CONNECTION_STATES.DISCONNECTED,
    inputSource = INPUT_SOURCES.DUMMY_VIDEO,
    uptime = '00:00:00'
}) => {
    const getConnectionClass = () => {
        switch (connectionStatus) {
            case CONNECTION_STATES.CONNECTED:
                return 'status-connected';
            case CONNECTION_STATES.CONNECTING:
                return 'status-connecting';
            case CONNECTION_STATES.FAILED:
                return 'status-failed';
            default:
                return 'status-disconnected';
        }
    };

    const getConnectionLabel = () => {
        switch (connectionStatus) {
            case CONNECTION_STATES.CONNECTED:
                return 'LIVE';
            case CONNECTION_STATES.CONNECTING:
                return 'CONNECTING';
            case CONNECTION_STATES.FAILED:
                return 'FAILED';
            default:
                return 'OFFLINE';
        }
    };

    const getStateBadgeClass = () => {
        switch (systemState) {
            case 'CONFIRMED_THREAT':
                return 'state-badge state-confirmed';
            case 'POTENTIAL_ANOMALY':
                return 'state-badge state-potential';
            default:
                return 'state-badge state-safe';
        }
    };

    /**
     * Get FPS value color class based on performance
     */
    const getFpsColorClass = (fps) => {
        if (fps >= 12) return 'value-green';
        if (fps >= 8) return 'value-yellow';
        return 'value-red';
    };

    /**
     * Get latency color class based on responsiveness
     */
    const getLatencyColorClass = (latency) => {
        if (latency === null) return 'value-white';
        if (latency <= 100) return 'value-green';
        if (latency <= 300) return 'value-yellow';
        return 'value-red';
    };

    /**
     * Get input source icon
     */
    const getInputSourceIcon = () => {
        switch (inputSource) {
            case INPUT_SOURCES.LIVE_CAMERA:
                return '📹';
            case INPUT_SOURCES.PHONE_CAMERA:
                return '📱';
            case INPUT_SOURCES.DUMMY_VIDEO:
            default:
                return '🎬';
        }
    };

    /**
     * Derive risk level from system state (frontend-only logic)
     * HIGH - SAFE MODE active (degraded system)
     * MODERATE - High latency (>300ms) or low FPS (<8)
     * LOW - Normal operation
     */
    const getRiskLevel = () => {
        if (systemState === 'SAFE_MODE') return 'HIGH';
        if ((latencyMs && latencyMs > 300) || renderFps < 8) return 'MODERATE';
        return 'LOW';
    };

    const getRiskLevelClass = () => {
        const level = getRiskLevel();
        switch (level) {
            case 'HIGH': return 'risk-badge risk-high';
            case 'MODERATE': return 'risk-badge risk-moderate';
            default: return 'risk-badge risk-low';
        }
    };

    return (
        <div className="status-bar">
            <div className="status-brand">
                <img
                    src="/logo.jpeg"
                    alt="Jal-Drishti Logo"
                    className="brand-logo"
                />
                <h1 className="brand-title">JAL-DRISHTI</h1>
                <span className="brand-version">PHASE-2</span>
            </div>

            <div className="status-metrics">
                {/* System State Badge */}
                <div className={getStateBadgeClass()}>
                    {STATE_LABELS[systemState] || 'SAFE MODE'}
                </div>

                {/* Risk Level Badge */}
                <div className={getRiskLevelClass()}>
                    RISK LEVEL: {getRiskLevel()}
                </div>

                {/* Max Confidence */}
                <div className="metric-group right-align">
                    <span className="metric-label">CONFIDENCE</span>
                    <span className="metric-value value-white">
                        {(maxConfidence * 100).toFixed(0)}%
                    </span>
                </div>

                {/* Latency - backend reported only */}
                <div className="metric-group right-align">
                    <span className="metric-label">LATENCY</span>
                    <span className={`metric-value ${getLatencyColorClass(latencyMs)}`}>
                        {latencyMs !== null ? `${latencyMs}ms` : '-- ms'}
                    </span>
                </div>

                {/* Render FPS (frontend) */}
                <div className="metric-group right-align">
                    <span className="metric-label">RENDER FPS</span>
                    <span className={`metric-value ${getFpsColorClass(renderFps)}`}>
                        {renderFps}
                    </span>
                </div>

                {/* ML FPS (backend, if available) */}
                {mlFps !== null && (
                    <div className="metric-group right-align">
                        <span className="metric-label">ML FPS</span>
                        <span className="metric-value value-cyan">{mlFps.toFixed(1)}</span>
                    </div>
                )}

                {/* Input Source Indicator */}
                <div className="input-source">
                    <span className="input-source-icon">{getInputSourceIcon()}</span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="input-source-label">Source</span>
                        <span className="input-source-value">{inputSource}</span>
                    </div>
                </div>

                {/* System Uptime */}
                <div className="uptime-display">
                    <span className="uptime-label">Uptime</span>
                    <span className="uptime-value">{uptime}</span>
                </div>

                {/* Connection Status */}
                <div className="metric-group">
                    <span className={`status-dot ${getConnectionClass()}`}></span>
                    <span className="status-text">{getConnectionLabel()}</span>
                </div>
            </div>
        </div>
    );
};

export default StatusBar;
