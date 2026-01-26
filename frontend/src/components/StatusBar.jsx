import React from 'react';
import { STATE_COLORS, STATE_LABELS, CONNECTION_STATES } from '../constants';
import '../App.css';

/**
 * StatusBar Component
 * 
 * Displays:
 * - System state badge (color-coded)
 * - Max confidence (from backend)
 * - Latency (backend-reported only, show -- if missing)
 * - Render FPS (frontend measured)
 * - ML FPS (backend-reported, if available)
 * - Connection status
 */
const StatusBar = ({
    systemState = 'SAFE_MODE',
    maxConfidence = 0,
    latencyMs = null,
    renderFps = 0,
    mlFps = null,
    connectionStatus = CONNECTION_STATES.DISCONNECTED
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
                return 'CONNECTED';
            case CONNECTION_STATES.CONNECTING:
                return 'CONNECTING';
            case CONNECTION_STATES.FAILED:
                return 'FAILED';
            default:
                return 'DISCONNECTED';
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

    return (
        <div className="status-bar">
            <div className="status-brand">
                <h1 className="brand-title">JAL-DRISHTI</h1>
                <span className="brand-version">v2.0.0-Phase2</span>
            </div>

            <div className="status-metrics">
                {/* System State Badge */}
                <div className={getStateBadgeClass()}>
                    {STATE_LABELS[systemState] || 'SAFE MODE'}
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
                    <span className="metric-value value-yellow">
                        {latencyMs !== null ? `${latencyMs}ms` : '-- ms'}
                    </span>
                </div>

                {/* Render FPS (frontend) */}
                <div className="metric-group right-align">
                    <span className="metric-label">RENDER FPS</span>
                    <span className="metric-value value-green">{renderFps}</span>
                </div>

                {/* ML FPS (backend, if available) */}
                {mlFps !== null && (
                    <div className="metric-group right-align">
                        <span className="metric-label">ML FPS</span>
                        <span className="metric-value value-cyan">{mlFps.toFixed(1)}</span>
                    </div>
                )}

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
