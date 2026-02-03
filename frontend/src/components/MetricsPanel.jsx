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
 */
const MetricsPanel = ({
    fpsHistory = [],
    latencyHistory = [],
    inSafeMode = false,
    safeModeStartTime = null
}) => {
    // Calculate safe mode duration if active
    const safeModeSeconds = inSafeMode && safeModeStartTime
        ? Math.floor((Date.now() - safeModeStartTime) / 1000)
        : 0;

    return (
        <div className="metrics-panel">
            <div className="metrics-panel-header">
                <span className="metrics-panel-title">📊 System Metrics</span>
            </div>

            <div className="metrics-panel-content">
                {/* FPS Graph */}
                <MetricsGraph
                    data={fpsHistory}
                    width={240}
                    height={55}
                    color="#00ff88"
                    label="FPS"
                    unit=""
                    maxPoints={60}
                    minValue={0}
                    maxValue={30}
                />

                {/* Latency Graph */}
                <MetricsGraph
                    data={latencyHistory}
                    width={240}
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
            </div>
        </div>
    );
};

export default MetricsPanel;
