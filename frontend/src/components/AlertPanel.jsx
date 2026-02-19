import React, { useState } from 'react';
import { SYSTEM_STATES } from '../constants';
import '../App.css';

/**
 * AlertPanel Component (v3 — Context-Driven)
 * 
 * Reads from centralized alertManager via SystemStateContext.
 * No local state tracking / debounce — all handled by alertManager.
 * Displays noise badges and SAFE_MODE reason codes.
 */
const AlertPanel = ({
    alerts = [],
    lowSignalReliability = false,
    safeModeReason = null,
    currentState = SYSTEM_STATES.SAFE_MODE,
    addEvent = null
}) => {
    const [handledAlertState, setHandledAlertState] = useState(null);

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
                    {lowSignalReliability && (
                        <span style={{
                            marginLeft: '8px',
                            background: '#F97316',
                            color: '#000',
                            padding: '1px 6px',
                            borderRadius: '3px',
                            fontSize: '9px',
                            fontWeight: 700
                        }}>
                            LOW SIGNAL
                        </span>
                    )}
                </div>
                <div className="header-status">
                    <span className="label">CURRENT STATUS:</span>
                    <span className={`status-value status-${currentState.toLowerCase().replace('_', '-')}`}>{currentState.replace('_', ' ')}</span>
                </div>
            </div>

            {/* Operator Actions Bar */}
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
                            <th width="12%">LEVEL</th>
                            <th width="43%">EVENT MESSAGE</th>
                            <th width="15%">CONFIDENCE</th>
                            <th width="15%">STATUS</th>
                        </tr>
                    </thead>
                    <tbody>
                        {alerts.length === 0 ? (
                            <tr className="empty-row">
                                <td colSpan="5">NO RECENT ALERTS LOGGED</td>
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
                                                style={{ width: `${(alert.confidence || 0) * 100}%` }}
                                            />
                                        </div>
                                        <span className="confidence-text">{((alert.confidence || 0) * 100).toFixed(0)}%</span>
                                    </td>
                                    <td style={{ fontSize: '9px' }}>
                                        {alert.badges && alert.badges.length > 0 && alert.badges.map((badge, i) => (
                                            <span key={i} style={{
                                                display: 'inline-block',
                                                background: badge === 'LOW SIGNAL RELIABILITY' ? '#F97316' : '#64748B',
                                                color: '#000',
                                                padding: '1px 4px',
                                                borderRadius: '2px',
                                                fontSize: '8px',
                                                fontWeight: 600,
                                                marginRight: '2px'
                                            }}>
                                                {badge}
                                            </span>
                                        ))}
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
