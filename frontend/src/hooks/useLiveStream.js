import { useState, useEffect, useRef, useCallback } from 'react';
import {
    SYSTEM_STATES,
    CONNECTION_STATES,
    RECONNECT_CONFIG,
    WS_CONFIG,
    SYSTEM_STATUS,
    EVENT_TYPES
} from '../constants';

/**
 * useLiveStream Hook
 * 
 * PHASE-3 OPTIMIZED:
 * - Frontend is PASSIVE subscriber
 * - Backend drives timing, not frontend
 * - Keep last valid frame on disconnect
 * - No pings to backend
 * - requestAnimationFrame decoupling: WS receive rate != render rate
 */
const useLiveStream = (enabled = true) => {
    const [frame, setFrame] = useState(null);
    const [fps, setFps] = useState(0);
    const [connectionStatus, setConnectionStatus] = useState(CONNECTION_STATES.DISCONNECTED);
    const [reconnectAttempt, setReconnectAttempt] = useState(0);

    // System status state for safe mode overlay
    const [systemStatus, setSystemStatus] = useState({
        inSafeMode: false,
        message: null,
        cause: null
    });

    // Event log for timeline
    const [events, setEvents] = useState([]);

    // Refs for logic that shouldn't trigger re-renders
    const wsRef = useRef(null);
    const frameCountRef = useRef(0);
    const fpsIntervalRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const lastValidFrameRef = useRef(null);
    const lastFrameIdRef = useRef(-1);
    const connectRef = useRef(null);

    // ========== requestAnimationFrame DECOUPLING ==========
    // Store latest frame in ref (fast, no re-render)
    // Only render via requestAnimationFrame (smooth, 60fps max)
    const latestFrameRef = useRef(null);
    const rafIdRef = useRef(null);
    const isRenderingRef = useRef(false);
    /**
     * Add event to timeline
     */
    const addEvent = useCallback((type, message, severity = 'info') => {
        const newEvent = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            }),
            type,
            message,
            severity // 'info', 'warning', 'danger', 'success'
        };
        setEvents(prev => [newEvent, ...prev].slice(0, 50)); // Keep last 50 events
    }, []);

    const getReconnectDelay = useCallback((attempt) => {
        return Math.min(
            RECONNECT_CONFIG.BASE_DELAY_MS * (2 ** attempt),
            RECONNECT_CONFIG.MAX_DELAY_MS
        );
    }, []);

    const cleanup = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (rafIdRef.current) {
            cancelAnimationFrame(rafIdRef.current);
            rafIdRef.current = null;
        }
    }, []);

    // ========== RENDER LOOP (Decoupled from WS) ==========
    // This runs at display refresh rate (~60fps) but only updates state
    // when there's a new frame in latestFrameRef
    const renderLoop = useCallback(() => {
        if (latestFrameRef.current && !isRenderingRef.current) {
            isRenderingRef.current = true;
            const frameToRender = latestFrameRef.current;
            latestFrameRef.current = null; // Clear buffer

            setFrame(frameToRender);
            lastValidFrameRef.current = frameToRender;
            frameCountRef.current += 1;

            isRenderingRef.current = false;
        }
        rafIdRef.current = requestAnimationFrame(renderLoop);
    }, []);

    /**
     * Handle system messages from backend
     * Message types: safe_mode, recovered, connected
     */
    const handleSystemMessage = useCallback((message) => {
        const { status, message: msg } = message;

        switch (status) {
            case SYSTEM_STATUS.SAFE_MODE:
                // Backend signaling safe mode entry
                setSystemStatus({
                    inSafeMode: true,
                    message: msg || 'System entered safe mode',
                    cause: message.cause || message.payload?.cause || 'Unknown cause'
                });
                addEvent(EVENT_TYPES.SAFE_MODE_ENTRY, msg || 'SAFE MODE Activated', 'danger');
                console.log('[WS] System entered SAFE MODE:', msg);
                break;

            case SYSTEM_STATUS.RECOVERED:
                // Backend signaling recovery from safe mode
                setSystemStatus({
                    inSafeMode: false,
                    message: null,
                    cause: null
                });
                addEvent(EVENT_TYPES.SAFE_MODE_EXIT, 'System Recovered', 'success');
                console.log('[WS] System RECOVERED from safe mode');
                break;

            case SYSTEM_STATUS.CONNECTED:
                // Initial connection confirmation
                addEvent(EVENT_TYPES.CONNECTION, 'WebSocket Connected', 'success');
                console.log('[WS] Connection confirmed by backend');
                break;

            default:
                console.log('[WS] Unknown system status:', status);
        }
    }, [addEvent]);

    // Connect to WebSocket function
    const connect = useCallback(() => {
        if (!enabled) return;

        try {
            const ws = new WebSocket(WS_CONFIG.URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[WS] Connected to AI Backend');
                setConnectionStatus(CONNECTION_STATES.CONNECTED);
                setReconnectAttempt(0);

                // Start render loop on connect
                if (!rafIdRef.current) {
                    rafIdRef.current = requestAnimationFrame(renderLoop);
                }
            };

            ws.onmessage = (event) => {
                try {
                    const response = JSON.parse(event.data);



                    // ==========================================
                    // SYSTEM MESSAGE HANDLING
                    // Handle type: "system" messages separately
                    // ==========================================
                    if (response.type === 'system') {
                        handleSystemMessage(response);
                        return;
                    }

                    // ==========================================
                    // DATA MESSAGE HANDLING
                    // Handle type: "data" messages (normal ML output)
                    // ==========================================

                    // For data messages, extract payload if wrapped
                    const data = response.payload || response;

                    // Out-of-order detection: ignore stale frames
                    if (data.frame_id !== undefined && data.frame_id <= lastFrameIdRef.current) {
                        return;
                    }
                    lastFrameIdRef.current = data.frame_id || lastFrameIdRef.current;

                    // Normalize image_data to a proper data URL if it's a base64 string
                    let imageData = data.image_data || null;
                    if (imageData && typeof imageData === 'string' && !imageData.startsWith('data:')) {
                        imageData = `data:image/jpeg;base64,${imageData}`;
                    }

                    const normalizedFrame = {
                        timestamp: data.timestamp || new Date().toISOString(),
                        state: data.state || SYSTEM_STATES.SAFE_MODE,
                        max_confidence: data.max_confidence ?? 0,
                        detections: data.detections || [],
                        visibility_score: data.visibility_score ?? 0,
                        image_data: imageData,
                        frame_id: data.frame_id,
                        system: {
                            fps: data.system?.fps ?? null,
                            latency_ms: data.system?.latency_ms ?? null
                        }
                    };

                    // DECOUPLING: Store in ref, don't call setFrame directly
                    // Render loop will pick it up on next animation frame
                    latestFrameRef.current = normalizedFrame;

                } catch (err) {
                    console.error('[WS] Error parsing response:', err);
                }
            };

            ws.onclose = (event) => {
                console.log('[WS] Connection closed:', event.code);
                addEvent(EVENT_TYPES.DISCONNECTION, `Connection closed (code: ${event.code})`, 'warning');

                if (streamIntervalRef.current) {
                    clearInterval(streamIntervalRef.current);
                    streamIntervalRef.current = null;
                }

                // Attempt reconnection if not at max attempts
                setReconnectAttempt((prev) => {
                    const nextAttempt = prev + 1;

                    if (nextAttempt >= RECONNECT_CONFIG.MAX_ATTEMPTS) {
                        console.log('[WS] Max reconnection attempts reached');
                        setConnectionStatus(CONNECTION_STATES.FAILED);
                        return nextAttempt;
                    }

                    setConnectionStatus(CONNECTION_STATES.DISCONNECTED);
                    const delay = getReconnectDelay(nextAttempt);
                    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${nextAttempt}/${RECONNECT_CONFIG.MAX_ATTEMPTS})`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        if (connectRef.current) {
                            connectRef.current();
                        }
                    }, delay);

                    return nextAttempt;
                });
            };

            ws.onerror = () => {
                // Error will be followed by onclose
            };

        } catch (err) {
            console.error('[WS] Failed to create WebSocket:', err);
            setConnectionStatus(CONNECTION_STATES.FAILED);
        }
    }, [getReconnectDelay, handleSystemMessage, addEvent]);

    // Keep the ref updated for timeout callbacks
    useEffect(() => {
        connectRef.current = connect;
    }, [connect]);


    // Manual reconnect (for retry button after FAILED state)
    const manualReconnect = useCallback(() => {
        console.log('[WS] Manual reconnect triggered');
        setReconnectAttempt(0);
        lastFrameIdRef.current = -1;
        addEvent(EVENT_TYPES.CONNECTION, 'Manual reconnect initiated', 'info');
        connect();
    }, [connect, addEvent]);

    // Initialize connection and FPS counter
    useEffect(() => {
        connect();

        // FPS calculation loop
        fpsIntervalRef.current = setInterval(() => {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
        }, 1000);

        return () => {
            cleanup();
            if (fpsIntervalRef.current) {
                clearInterval(fpsIntervalRef.current);
            }
        };
    }, [connect, cleanup]);

    return {
        frame,
        fps,
        connectionStatus,
        reconnectAttempt,
        lastValidFrame: lastValidFrameRef.current,
        manualReconnect,
        // New exports for UI enhancement
        systemStatus,
        events,
        addEvent
    };
};

export default useLiveStream;
