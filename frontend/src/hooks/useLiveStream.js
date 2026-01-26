import { useState, useEffect, useRef, useCallback } from 'react';
import {
    SYSTEM_STATES,
    CONNECTION_STATES,
    RECONNECT_CONFIG,
    WS_CONFIG
} from '../constants';

const useLiveStream = (token) => {
    const [frame, setFrame] = useState(null);
    const [fps, setFps] = useState(0);
    const [connectionStatus, setConnectionStatus] = useState(CONNECTION_STATES.DISCONNECTED);
    const [reconnectAttempt, setReconnectAttempt] = useState(0);

    // Refs for logic that shouldn't trigger re-renders
    const wsRef = useRef(null);
    const frameCountRef = useRef(0);
    const fpsIntervalRef = useRef(null);
    const streamIntervalRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const lastValidFrameRef = useRef(null);
    const lastFrameIdRef = useRef(-1);
    const connectRef = useRef(null);

    useEffect(() => {
        // Connect to WebSocket
        if (!token) return;

        const connect = () => {
            const ws = new WebSocket(`${BACKEND_URL}?token=${token}`);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[WS] Connected to AI Backend');
                setConnectionStatus(CONNECTION_STATES.CONNECTED);
                setReconnectAttempt(0);

                // Start sending frames to drive the backend loop
                streamIntervalRef.current = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(new Uint8Array([0]));
                    }
                }, WS_CONFIG.FRAME_INTERVAL_MS);
            };

            ws.onmessage = (event) => {
                try {
                    const response = JSON.parse(event.data);

                    // Out-of-order detection: ignore stale frames
                    if (response.frame_id !== undefined && response.frame_id <= lastFrameIdRef.current) {
                        return;
                    }
                    lastFrameIdRef.current = response.frame_id || lastFrameIdRef.current;

                    // Normalize response to Phase-2 schema with safe defaults
                    const normalizedFrame = {
                        timestamp: response.timestamp || new Date().toISOString(),
                        state: response.state || SYSTEM_STATES.SAFE_MODE,
                        max_confidence: response.max_confidence ?? 0,
                        detections: response.detections || [],
                        visibility_score: response.visibility_score ?? 0,
                        image_data: response.image_data || null,
                        frame_id: response.frame_id,
                        system: {
                            fps: response.system?.fps ?? null,
                            latency_ms: response.system?.latency_ms ?? null
                        }
                    };

                    setFrame(normalizedFrame);
                    lastValidFrameRef.current = normalizedFrame;
                    frameCountRef.current += 1;

                } catch (err) {
                    console.error('[WS] Error parsing response:', err);
                }
            };

            ws.onclose = (event) => {
                console.log('[WS] Connection closed:', event.code);
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
    }, [cleanup, getReconnectDelay]);

    // Store connect in ref for use in callbacks
    connectRef.current = connect;

    // Manual reconnect (for retry button after FAILED state)
    const manualReconnect = useCallback(() => {
        console.log('[WS] Manual reconnect triggered');
        setReconnectAttempt(0);
        lastFrameIdRef.current = -1;
        connect();
    }, [connect]);

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
    }, [token]);

    return {
        frame,
        fps,
        connectionStatus,
        reconnectAttempt,
        lastValidFrame: lastValidFrameRef.current,
        manualReconnect
    };
};

export default useLiveStream;
