import React, { useRef, useEffect, useState } from 'react';
import { SYSTEM_STATES, STATE_COLORS } from '../constants';
import '../App.css';

/**
 * AlertPanel Component (Redesigned Phase-3)
 * 
 * High-tech, analytical display of system alerts.
 * Features:
 * - Tabular layout for clean data scanning
 * - Status indicators with pulse effects
 * - Clear action buttons for operator
 * - Concise, professional typography
 */
const AlertPanel = ({
    currentState = SYSTEM_STATES.SAFE_MODE,
    detections = [],
    maxConfidence = 0,
    addEvent = null
}) => {
    const prevStateRef = useRef(currentState);
    const [alerts, setAlerts] = useState([]);
    const [handledAlertState, setHandledAlertState] = useState(null);

    // Handle state transitions
    useEffect(() => {
        const prevState = prevStateRef.current;
        if (prevState !== currentState) {
            const timestamp = new Date().toLocaleTimeString();
            let newAlert = null;

            if (prevState === SYSTEM_STATES.SAFE_MODE && currentState === SYSTEM_STATES.POTENTIAL_ANOMALY) {
                newAlert = {
                    id: Date.now(),
                    type: 'WARNING',
                    message: 'POTENTIAL ANOMALY DETECTED',
                    timestamp,
                    confidence: maxConfidence
                };
            } else if (prevState === SYSTEM_STATES.POTENTIAL_ANOMALY && currentState === SYSTEM_STATES.CONFIRMED_THREAT) {
                newAlert = {
                    id: Date.now(),
                    type: 'CRITICAL',
                    message: 'THREAT CONFIRMED - ACTION REQUIRED',
                    timestamp,
                    confidence: maxConfidence
                };
            } else if (
                (prevState === SYSTEM_STATES.CONFIRMED_THREAT || prevState === SYSTEM_STATES.POTENTIAL_ANOMALY) &&
                currentState === SYSTEM_STATES.SAFE_MODE
            ) {
                newAlert = {
                    id: Date.now(),
                    type: 'INFO',
                    message: 'THREAT CLEARED / NORMAL OPERATIONS',
                    timestamp,
                    confidence: maxConfidence
                };
            }

            if (newAlert) {
                setAlerts((prev) => [newAlert, ...prev].slice(0, 15));
            }
            setHandledAlertState(null);
            prevStateRef.current = currentState;
        }
    }, [currentState, maxConfidence]);

    const getRowClass = (type) => {
        switch (type) {
            case 'CRITICAL': return 'row-critical';
            case 'WARNING': return 'row-warning';
            case 'INFO': return 'row-info';
            default: return '';
        }
    };

    const handleConfirmThreat = () => {
        if (addEvent) addEvent('manual_action', 'OPERATOR CONFIRMED THREAT', 'danger');
        setHandledAlertState('confirmed');
    };

    const handleDismiss = () => {
        if (addEvent) addEvent('manual_action', 'OPERATOR DISMISSED ALARM', 'success');
        setHandledAlertState('dismissed');
    };

    const isAlertActive = currentState === SYSTEM_STATES.POTENTIAL_ANOMALY || currentState === SYSTEM_STATES.CONFIRMED_THREAT;
    const showActions = isAlertActive && handledAlertState === null;

    return (
        <div className="alert-panel-container">
            {/* Header / Current Status */}
            <div className="alert-panel-header">
                <div className="header-title">
                    <span className="icon">⚠️</span>
                    SYSTEM ALERT LOG
                </div>
                <div className="header-status">
                    <span className="label">CURRENT STATUS:</span>
                    <span className={`status-value status-${currentState.toLowerCase().replace('_', '-')}`}>
                        {currentState.replace('_', ' ')}
                    </span>
                </div>
            </div>

            {/* Operator Actions Bar - Only visible when needed */}
            {showActions && (
                <div className="operator-actions-bar">
                    <span className="action-label">OPERATOR DECISION REQUIRED:</span>
                    <div className="action-buttons">
                        <button className="btn-action btn-confirm" onClick={handleConfirmThreat}>
                            CONFIRM THREAT
                        </button>
                        <button className="btn-action btn-dismiss" onClick={handleDismiss}>
                            DISMISS FALSE ALARM
                        </button>
                    </div>
                </div>
            )}

            {/* Analytical Log Table */}
            <div className="alert-table-wrapper custom-scrollbar">
                <table className="alert-table">
                    <thead>
                        <tr>
                            <th width="15%">TIME</th>
                            <th width="15%">LEVEL</th>
                            <th width="50%">EVENT MESSAGE</th>
                            <th width="20%">CONFIDENCE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {alerts.length === 0 ? (
                            <tr className="empty-row">
                                <td colSpan="4">NO RECENT ALERTS LOGGED</td>
                            </tr>
                        ) : (
                            alerts.map((alert) => (
                                <tr key={alert.id} className={getRowClass(alert.type)}>
                                    <td className="timestamp">{alert.timestamp}</td>
                                    <td className="level-badge"><span className="badge">{alert.type}</span></td>
                                    <td className="message">{alert.message}</td>
                                    <td className="confidence">
                                        <div className="confidence-bar-bg">
                                            <div
                                                className="confidence-bar-fill"
                                                style={{ width: `${alert.confidence * 100}%` }}
                                            />
                                        </div>
                                        <span className="confidence-text">{(alert.confidence * 100).toFixed(0)}%</span>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default AlertPanel;
