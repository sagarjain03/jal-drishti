import React, { useRef, useEffect, useState } from 'react';
import { SYSTEM_STATES, STATE_COLORS } from '../constants';
import '../App.css';

/**
 * AlertPanel Component
 * 
 * State-transition-driven alerts (never per-frame).
 * Only shows messages on state changes:
 * - SAFE → POTENTIAL: "Potential anomaly detected"
 * - POTENTIAL → CONFIRMED: Escalated alert
 * - CONFIRMED → SAFE: "Threat cleared"
 * - Any → SAFE_MODE: "Low confidence / poor visibility"
 */
const AlertPanel = ({
    currentState = SYSTEM_STATES.SAFE_MODE,
    detections = [],
    maxConfidence = 0
}) => {
    const prevStateRef = useRef(currentState);
    const [alerts, setAlerts] = useState([]);

    // Handle state transitions
    useEffect(() => {
        const prevState = prevStateRef.current;

        if (prevState !== currentState) {
            const timestamp = new Date().toLocaleTimeString();
            let newAlert = null;

            // Determine transition message
            if (prevState === SYSTEM_STATES.SAFE_MODE && currentState === SYSTEM_STATES.POTENTIAL_ANOMALY) {
                newAlert = {
                    id: Date.now(),
                    type: 'warning',
                    message: 'Potential anomaly detected',
                    timestamp,
                    confidence: maxConfidence
                };
            } else if (prevState === SYSTEM_STATES.POTENTIAL_ANOMALY && currentState === SYSTEM_STATES.CONFIRMED_THREAT) {
                newAlert = {
                    id: Date.now(),
                    type: 'danger',
                    message: 'THREAT CONFIRMED - Immediate attention required',
                    timestamp,
                    confidence: maxConfidence
                };
            } else if (
                (prevState === SYSTEM_STATES.CONFIRMED_THREAT || prevState === SYSTEM_STATES.POTENTIAL_ANOMALY) &&
                currentState === SYSTEM_STATES.SAFE_MODE
            ) {
                newAlert = {
                    id: Date.now(),
                    type: 'success',
                    message: 'Threat cleared / visibility improved',
                    timestamp,
                    confidence: maxConfidence
                };
            } else if (currentState === SYSTEM_STATES.SAFE_MODE && prevState !== SYSTEM_STATES.SAFE_MODE) {
                newAlert = {
                    id: Date.now(),
                    type: 'info',
                    message: 'Low confidence / poor visibility',
                    timestamp,
                    confidence: maxConfidence
                };
            }

            if (newAlert) {
                setAlerts((prev) => [newAlert, ...prev].slice(0, 10)); // Keep last 10 alerts
            }

            prevStateRef.current = currentState;
        }
    }, [currentState, maxConfidence]);

    const getAlertClass = (type) => {
        switch (type) {
            case 'danger': return 'alert-item-danger';
            case 'warning': return 'alert-item-warning';
            case 'success': return 'alert-item-success';
            default: return 'alert-item-info';
        }
    };

    const getStateMessage = () => {
        switch (currentState) {
            case SYSTEM_STATES.CONFIRMED_THREAT:
                return 'THREAT DETECTED';
            case SYSTEM_STATES.POTENTIAL_ANOMALY:
                return 'MONITORING ANOMALY';
            default:
                return 'No active threats detected.';
        }
    };

    return (
        <div className="alert-panel">
            <div className="alert-header">
                <h3 className="alert-title">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ marginRight: '8px', color: STATE_COLORS[currentState] }}>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    System Alerts
                </h3>
                <span className="alert-count">{alerts.length} Events</span>
            </div>

            <div className="alert-list custom-scrollbar">
                {/* Current State Banner */}
                <div className={`state-banner state-banner-${currentState.toLowerCase()}`}>
                    {getStateMessage()}
                </div>

                {/* Alert History */}
                {alerts.length === 0 ? (
                    <div className="alert-empty">No state transitions recorded.</div>
                ) : (
                    alerts.map((alert) => (
                        <div key={alert.id} className={`alert-item ${getAlertClass(alert.type)}`}>
                            <div className="alert-info">
                                <span className="alert-time">{alert.timestamp}</span>
                                <span className="alert-message">{alert.message}</span>
                            </div>
                            <div className="alert-meta">
                                <span className="alert-confidence">
                                    {(alert.confidence * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default AlertPanel;
