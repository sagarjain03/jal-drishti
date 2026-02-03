import React from 'react';
import '../App.css';

/**
 * LastAlertSnapshot Component
 * 
 * Compact panel showing the last automatically captured alert snapshot.
 * Features:
 * - Auto-capture on alert state
 * - Timestamp + alert label display
 * - "No alerts recorded" placeholder
 */
const LastAlertSnapshot = ({ snapshot }) => {
    const getAlertTypeClass = () => {
        if (!snapshot) return '';
        switch (snapshot.alertType) {
            case 'CONFIRMED_THREAT': return 'threat';
            case 'POTENTIAL_ANOMALY': return 'anomaly';
            default: return '';
        }
    };

    const getAlertTypeLabel = () => {
        if (!snapshot) return '';
        switch (snapshot.alertType) {
            case 'CONFIRMED_THREAT': return 'THREAT';
            case 'POTENTIAL_ANOMALY': return 'ANOMALY';
            default: return 'ALERT';
        }
    };

    return (
        <div className="last-alert-panel">
            <div className="last-alert-header">
                <span className="last-alert-title">
                    🚨 Last Alert Snapshot
                </span>
            </div>

            <div className="last-alert-content">
                {!snapshot || !snapshot.imageData ? (
                    <div className="last-alert-empty">
                        No alerts recorded
                    </div>
                ) : (
                    <div className="last-alert-snapshot">
                        <img
                            src={snapshot.imageData}
                            alt="Last Alert"
                            className="last-alert-image"
                        />
                        <div className="last-alert-info">
                            <span className="last-alert-time">{snapshot.timestamp}</span>
                            <span className={`last-alert-type ${getAlertTypeClass()}`}>
                                {getAlertTypeLabel()}
                            </span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default LastAlertSnapshot;
