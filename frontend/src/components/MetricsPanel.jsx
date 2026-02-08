import React from 'react';
import MetricsGraph from './MetricsGraph';
import '../App.css';

/**
 * MetricsPanel Component
 * 
 * Container showing real-time system metrics:
 * - FPS over time (last 60 seconds)
 * - ML Latency over time
 * - Safe Mode indicator timeline
 * - Situational Awareness Score (system-level)
 */
const MetricsPanel = ({
    fpsHistory = [],
    latencyHistory = [],
    inSafeMode = false,
    safeModeStartTime = null,
    // New props for awareness score
    currentFps = 0,
    latency = null,
    connectionStatus = 'disconnected',
    systemState = 'SAFE_MODE'
}) => {
    // Calculate safe mode duration if active
    const safeModeSeconds = inSafeMode && safeModeStartTime
        ? Math.floor((Date.now() - safeModeStartTime) / 1000)
        : 0;

    /**
     * Calculate Situational Awareness Score (0-100)
     * 
     * This is a system-level score based on observable metrics:
     * - Stable FPS (>= 12): +20
     * - Low latency (<= 100ms): +20
     * - Connected state: +20
     * - SAFE MODE inactive: +20
     * - No confirmed threat: +20
     * 
     * NOTE: This is NOT an ML confidence score.
     * It represents system health and readiness.
     */
    const calculateAwarenessScore = () => {
        let score = 0;

        // Stable FPS (>= 12)
        if (currentFps >= 12) score += 20;
        else if (currentFps >= 8) score += 10;

        // Low latency (<= 100ms)
        if (latency !== null && latency <= 100) score += 20;
        else if (latency !== null && latency <= 200) score += 10;

        // Connected state
        if (connectionStatus === 'connected') score += 20;

        // SAFE MODE inactive (normal operation)
        if (!inSafeMode) score += 20;

        // No confirmed threat active
        if (systemState !== 'CONFIRMED_THREAT') score += 20;
        else if (systemState === 'POTENTIAL_ANOMALY') score += 10;

        return score;
    };

    const awarenessScore = calculateAwarenessScore();

    const getAwarenessClass = () => {
        if (awarenessScore >= 80) return 'awareness-high';
        if (awarenessScore >= 50) return 'awareness-medium';
        return 'awareness-low';
    };


    return (
        <div className="metrics-panel">
            <div className="metrics-panel-header">
                <span className="metrics-panel-title">📊 System Metrics</span>
            </div>

            <div className="metrics-panel-content">
                {/* FPS Graph */}
                <MetricsGraph
                    data={fpsHistory}
                    width={400}
                    height={55}
                    color="#22C55E"
                    label="FPS"
                    unit=""
                    maxPoints={60}
                    minValue={0}
                    maxValue={30}
                />

                {/* Latency Graph */}
                <MetricsGraph
                    data={latencyHistory}
                    width={400}
                    height={55}
                    color="#00d4ff"
                    label="Latency"
                    unit="ms"
                    maxPoints={60}
                    minValue={0}
                />

                {/* Safe Mode Status */}
                <div className={`safe-mode-status ${inSafeMode ? 'active' : ''}`}>
                    <div className="safe-mode-indicator">
                        <span className={`safe-mode-dot ${inSafeMode ? 'danger' : 'success'}`}></span>
                        <span className="safe-mode-label">
                            {inSafeMode ? 'SAFE MODE' : 'NORMAL'}
                        </span>
                    </div>
                    {inSafeMode && (
                        <span className="safe-mode-duration">
                            {safeModeSeconds}s
                        </span>
                    )}
                </div>

                {/* Situational Awareness Score */}
                <div className="awareness-score">
                    <span className="awareness-score-label">Situational Awareness</span>
                    <span className={`awareness-score-value ${getAwarenessClass()}`}>
                        {awarenessScore} / 100
                    </span>
                </div>
            </div>
        </div>

    );
};

export default MetricsPanel;
