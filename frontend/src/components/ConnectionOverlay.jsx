import React from 'react';
import { CONNECTION_STATES, RECONNECT_CONFIG, OVERLAY_OPACITY } from '../constants';
import '../App.css';

/**
 * ConnectionOverlay Component
 * 
 * Non-blocking overlay that appears on connection issues.
 * - Canvas remains mounted (last frame frozen)
 * - Reconnecting: 50% opacity
 * - FAILED: 70% opacity (visually distinct)
 * - Manual retry button after FAILED state
 */
const ConnectionOverlay = ({
    connectionStatus,
    reconnectAttempt,
    onRetry
}) => {
    // Only show overlay when not connected
    if (connectionStatus === CONNECTION_STATES.CONNECTED) {
        return null;
    }

    const isFailed = connectionStatus === CONNECTION_STATES.FAILED;
    const isConnecting = connectionStatus === CONNECTION_STATES.CONNECTING;
    const opacity = isFailed ? OVERLAY_OPACITY.FAILED : OVERLAY_OPACITY.RECONNECTING;

    return (
        <div
            className="connection-overlay"
            style={{ backgroundColor: `rgba(0, 0, 0, ${opacity})` }}
        >
            <div className="connection-overlay-content">
                {isFailed ? (
                    <>
                        <div className="connection-icon connection-failed">⚠</div>
                        <h2 className="connection-title">DISCONNECTED</h2>
                        <p className="connection-message">
                            Unable to connect to backend after {RECONNECT_CONFIG.MAX_ATTEMPTS} attempts.
                        </p>
                        <p className="connection-instruction">
                            CHECK BACKEND STATUS
                        </p>
                        <button className="retry-button" onClick={onRetry}>
                            Retry Connection
                        </button>
                    </>
                ) : isConnecting ? (
                    <>
                        <div className="connection-icon connection-connecting">⟳</div>
                        <h2 className="connection-title">CONNECTING</h2>
                        <p className="connection-message">
                            Establishing connection to AI backend...
                        </p>
                    </>
                ) : (
                    <>
                        <div className="connection-icon connection-reconnecting">⟳</div>
                        <h2 className="connection-title">RECONNECTING</h2>
                        <p className="connection-message">
                            Attempt {reconnectAttempt} of {RECONNECT_CONFIG.MAX_ATTEMPTS}
                        </p>
                        <p className="connection-instruction">
                            Last frame preserved. Connection will resume automatically.
                        </p>
                    </>
                )}
            </div>
        </div>
    );
};

export default ConnectionOverlay;
