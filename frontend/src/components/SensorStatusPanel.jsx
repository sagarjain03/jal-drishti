import React from 'react';
import { FUSION_STATES, FUSION_STATE_COLORS, FUSION_STATE_LABELS, SENSOR_ROLES } from '../constants';
import '../App.css';

/**
 * SensorStatusPanel Component (MILESTONE-1)
 * 
 * Displays layered sensor intelligence:
 * - Sonar status with range + confidence
 * - IR status with confidence
 * - Camera status (final confirmation)
 * - Fusion state with timeline messages
 * 
 * Key principle: Camera is NOT the primary detector
 */
const SensorStatusPanel = ({
    sensors = { sonar: {}, ir: {}, camera: {} },
    fusionState = 'NORMAL',
    fusionMessage = '',
    timelineMessages = []
}) => {
    // Extract sensor data with defaults
    const sonar = sensors.sonar || { detected: false, distance_m: 0, confidence: 0 };
    const ir = sensors.ir || { detected: false, confidence: 0 };
    const camera = sensors.camera || { detected: false, confidence: 0, ml_available: true };

    const getFusionStateColor = () => FUSION_STATE_COLORS[fusionState] || FUSION_STATE_COLORS.NORMAL;
    const getFusionStateLabel = () => FUSION_STATE_LABELS[fusionState] || 'NORMAL';

    /**
     * Get sensor indicator class based on detection status
     */
    const getSensorClass = (detected, available = true) => {
        if (!available) return 'sensor-offline';
        return detected ? 'sensor-active' : 'sensor-idle';
    };

    /**
     * Format confidence as percentage
     */
    const formatConfidence = (conf) => {
        return conf > 0 ? `${(conf * 100).toFixed(0)}%` : '--';
    };

    return (
        <div className="sensor-status-panel">
            {/* Header with Fusion State */}
            <div className="sensor-panel-header">
                <div className="header-left">
                    <span className="sensor-icon">📡</span>
                    <span className="header-title">SENSOR ARRAY</span>
                </div>
                <div
                    className="fusion-state-badge"
                    style={{ backgroundColor: getFusionStateColor() }}
                >
                    {getFusionStateLabel()}
                </div>
            </div>

            {/* Sensor Status Grid */}
            <div className="sensor-grid">
                {/* Sonar Sensor */}
                <div className={`sensor-card ${getSensorClass(sonar.detected)}`}>
                    <div className="sensor-header">
                        <span className="sensor-emoji">🔊</span>
                        <span className="sensor-name">SONAR</span>
                        <span className={`sensor-dot ${sonar.detected ? 'active' : ''}`}></span>
                    </div>
                    <div className="sensor-role">{SENSOR_ROLES.SONAR.role}</div>
                    <div className="sensor-data">
                        <div className="data-row">
                            <span className="data-label">Range</span>
                            <span className="data-value">
                                {sonar.detected ? `${sonar.distance_m}m` : '---'}
                            </span>
                        </div>
                        <div className="data-row">
                            <span className="data-label">Conf</span>
                            <span className="data-value">{formatConfidence(sonar.confidence)}</span>
                        </div>
                    </div>
                    {sonar.detected && (
                        <div className="sensor-confidence-bar">
                            <div
                                className="confidence-fill sonar-fill"
                                style={{ width: `${sonar.confidence * 100}%` }}
                            />
                        </div>
                    )}
                </div>

                {/* IR Sensor */}
                <div className={`sensor-card ${getSensorClass(ir.detected)}`}>
                    <div className="sensor-header">
                        <span className="sensor-emoji">🌡️</span>
                        <span className="sensor-name">IR/THERMAL</span>
                        <span className={`sensor-dot ${ir.detected ? 'active' : ''}`}></span>
                    </div>
                    <div className="sensor-role">{SENSOR_ROLES.IR.role}</div>
                    <div className="sensor-data">
                        <div className="data-row">
                            <span className="data-label">Status</span>
                            <span className="data-value">
                                {ir.detected ? 'DETECTED' : 'CLEAR'}
                            </span>
                        </div>
                        <div className="data-row">
                            <span className="data-label">Conf</span>
                            <span className="data-value">{formatConfidence(ir.confidence)}</span>
                        </div>
                    </div>
                    {ir.detected && (
                        <div className="sensor-confidence-bar">
                            <div
                                className="confidence-fill ir-fill"
                                style={{ width: `${ir.confidence * 100}%` }}
                            />
                        </div>
                    )}
                </div>

                {/* Camera Sensor */}
                <div className={`sensor-card ${getSensorClass(camera.detected, camera.ml_available)}`}>
                    <div className="sensor-header">
                        <span className="sensor-emoji">📷</span>
                        <span className="sensor-name">CAMERA</span>
                        <span className={`sensor-dot ${camera.detected ? 'active' : ''} ${!camera.ml_available ? 'offline' : ''}`}></span>
                    </div>
                    <div className="sensor-role">{SENSOR_ROLES.CAMERA.role}</div>
                    <div className="sensor-data">
                        <div className="data-row">
                            <span className="data-label">Status</span>
                            <span className="data-value">
                                {!camera.ml_available ? 'OFFLINE' : camera.detected ? 'CONFIRMED' : 'MONITORING'}
                            </span>
                        </div>
                        <div className="data-row">
                            <span className="data-label">Conf</span>
                            <span className="data-value">{formatConfidence(camera.confidence)}</span>
                        </div>
                    </div>
                    {camera.detected && camera.ml_available && (
                        <div className="sensor-confidence-bar">
                            <div
                                className="confidence-fill camera-fill"
                                style={{ width: `${camera.confidence * 100}%` }}
                            />
                        </div>
                    )}
                </div>
            </div>

            {/* Fusion Message */}
            {fusionMessage && (
                <div className="fusion-message">
                    <span className="message-icon">ℹ️</span>
                    <span className="message-text">{fusionMessage}</span>
                </div>
            )}

            {/* Timeline Messages */}
            {timelineMessages && timelineMessages.length > 0 && (
                <div className="sensor-timeline">
                    {timelineMessages.map((msg, idx) => (
                        <div key={idx} className="timeline-item">
                            <span className="timeline-dot"></span>
                            <span className="timeline-text">{msg}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Info Footer */}
            <div className="sensor-footer">
                <span className="footer-note">
                    Camera = Final Confirmation Only
                </span>
            </div>
        </div>
    );
};

export default SensorStatusPanel;
