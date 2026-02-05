import React, { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE_URL, SOURCE_STATES } from '../constants';
import '../App.css';

/**
 * InputSourceToggle Component
 * 
 * PHASE-3 FIX: Optimistic UI + Source Status Polling
 * - Updates UI immediately on click (optimistic)
 * - Polls backend for reconciliation (every 1s)
 * - No page reload required
 */
const InputSourceToggle = ({ currentSource, onToggle, sourceState, onReset }) => {
    const [serverInfo, setServerInfo] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // PHASE-3 FIX: Optimistic UI state (local, instant)
    const [optimisticState, setOptimisticState] = useState(null);
    const pollingRef = useRef(null);

    const isCamera = currentSource === 'camera';

    // Use optimistic state if set, otherwise use prop
    const displayState = optimisticState || sourceState;
    const isWaiting = displayState === SOURCE_STATES.CAMERA_WAITING;
    const isIdle = displayState === SOURCE_STATES.IDLE;

    // Fetch server info on mount
    useEffect(() => {
        const fetchServerInfo = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/source/info`);
                if (res.ok) {
                    const data = await res.json();
                    setServerInfo(data);
                }
            } catch (err) {
                console.warn('[InputSourceToggle] Failed to fetch server info:', err);
            }
        };
        fetchServerInfo();
    }, []);

    // PHASE-3 FIX: Poll backend status for reconciliation
    useEffect(() => {
        const pollStatus = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/source/status`);
                if (res.ok) {
                    const data = await res.json();
                    const backendState = data.state;

                    // Reconcile: if optimistic state differs from backend, update
                    if (optimisticState && optimisticState !== backendState) {
                        console.log(`[InputSourceToggle] Reconciling: ${optimisticState} -> ${backendState}`);
                        setOptimisticState(null); // Clear optimistic, trust backend
                    }

                    // Update parent with backend state
                    if (backendState !== sourceState) {
                        const sourceType = data.source || (backendState.includes('VIDEO') ? 'video' : 'camera');
                        onToggle(sourceType, backendState);
                    }
                }
            } catch (err) {
                // Silent fail - will retry on next poll
            }
        };

        // Start polling every 1 second
        pollingRef.current = setInterval(pollStatus, 1000);

        // Initial poll
        pollStatus();

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
            }
        };
    }, [optimisticState, sourceState, onToggle]);

    // Handle source toggle with OPTIMISTIC UI
    const handleToggle = useCallback(async (sourceType) => {
        if (isLoading) return;

        setIsLoading(true);
        setError(null);

        // PHASE-3 FIX: Optimistic update IMMEDIATELY
        const optimisticNewState = sourceType === 'camera'
            ? SOURCE_STATES.CAMERA_WAITING
            : SOURCE_STATES.VIDEO_ACTIVE;
        setOptimisticState(optimisticNewState);

        // Update parent immediately for instant UI feedback
        onToggle(sourceType, optimisticNewState);
        if (onReset) onReset();

        try {
            const res = await fetch(`${API_BASE_URL}/api/source/select`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: sourceType })
            });

            const data = await res.json();

            if (data.success) {
                // Backend confirmed - clear optimistic (polling will reconcile)
                console.log(`[InputSourceToggle] Backend confirmed: ${data.state}`);
            } else {
                // Backend failed - revert optimistic
                setError(data.error || 'Failed to switch source');
                setOptimisticState(null);
            }
        } catch (err) {
            setError('Network error');
            setOptimisticState(null);
            console.error('[InputSourceToggle] API error:', err);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, onToggle, onReset]);

    // Get status text based on state
    const getStatusText = () => {
        if (isLoading) return 'SWITCHING...';
        if (isWaiting) return 'WAITING FOR PHONE';
        if (displayState === SOURCE_STATES.VIDEO_ACTIVE) return 'VIDEO ACTIVE';
        if (displayState === SOURCE_STATES.CAMERA_ACTIVE) return 'CAMERA ACTIVE';
        if (displayState === SOURCE_STATES.ERROR) return 'ERROR';
        if (isIdle) return 'SELECT SOURCE';
        return isCamera ? 'CAMERA' : 'VIDEO';
    };

    // Get status color class
    const getStatusClass = () => {
        if (isLoading || isWaiting) return 'status-waiting';
        if (displayState === SOURCE_STATES.VIDEO_ACTIVE) return 'status-video';
        if (displayState === SOURCE_STATES.CAMERA_ACTIVE) return 'status-camera';
        if (displayState === SOURCE_STATES.ERROR) return 'status-error';
        if (isIdle) return 'status-idle';
        return isCamera ? 'status-camera' : 'status-video';
    };

    return (
        <div className="input-source-toggle-container">
            <div className="toggle-header">
                <span className="toggle-label">INPUT SOURCE CONTROL</span>
                <span className={`toggle-status ${getStatusClass()}`}>
                    {getStatusText()}
                </span>
            </div>

            <div className="toggle-controls">
                <button
                    className={`source-btn ${!isCamera && !isIdle ? 'active' : ''} ${isLoading ? 'disabled' : ''}`}
                    onClick={() => handleToggle('video')}
                    disabled={isLoading}
                    title="Use Pre-recorded Video"
                >
                    <span className="source-icon">🎬</span>
                    <div className="source-info">
                        <span className="source-name">VIDEO FILE</span>
                        <span className="source-detail">RTSP / MP4</span>
                    </div>
                </button>

                <div className="toggle-divider"></div>

                <button
                    className={`source-btn ${isCamera ? 'active' : ''} ${isLoading ? 'disabled' : ''}`}
                    onClick={() => handleToggle('camera')}
                    disabled={isLoading}
                    title="Use Phone Camera"
                >
                    <span className="source-icon">📱</span>
                    <div className="source-info">
                        <span className="source-name">PHONE CAM</span>
                        <span className="source-detail">LIVE STREAM</span>
                    </div>
                </button>
            </div>

            {/* Camera URL Box - shown when camera is selected */}
            {isCamera && serverInfo && (
                <div className="camera-url-box">
                    <span className="url-label">📱 PHONE URL:</span>
                    <a
                        href={serverInfo.camera_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="camera-url"
                    >
                        {serverInfo.camera_url}
                    </a>
                    <span className="url-hint">Open on phone to stream</span>
                </div>
            )}

            {/* Error display */}
            {error && (
                <div className="toggle-error">
                    ⚠️ {error}
                </div>
            )}

            <div className="toggle-footer">
                <div className="signal-strength">
                    <span>STATE:</span>
                    <span className={`state-badge ${isIdle ? 'idle' : 'active'}`}>
                        {displayState || 'UNKNOWN'}
                    </span>
                </div>
                <span className="source-id">
                    IP: {serverInfo?.ip || '...'}
                </span>
            </div>
        </div>
    );
};

export default InputSourceToggle;
