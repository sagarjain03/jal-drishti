import { useState, useEffect, useRef } from 'react';
import { generateNextFrame } from '../data/mockStreamGenerator';
import { CONNECTION_STATES } from '../constants';

/**
 * useFakeStream Hook
 * 
 * Simulates the live stream for frontend-only testing.
 * Generates Phase-2 compliant frames with all required fields.
 */
const useFakeStream = () => {
    const [frame, setFrame] = useState(null);
    const [fps, setFps] = useState(0);
    const [connectionStatus, setConnectionStatus] = useState(CONNECTION_STATES.DISCONNECTED);
    const [reconnectAttempt, setReconnectAttempt] = useState(0);

    // Refs for FPS calculation
    const frameCountRef = useRef(0);
    const intervalRef = useRef(null);
    const fpsIntervalRef = useRef(null);
    const lastValidFrameRef = useRef(null);

    useEffect(() => {
        // Simulate connection startup
        setConnectionStatus(CONNECTION_STATES.CONNECTING);

        const startupTimeout = setTimeout(() => {
            setConnectionStatus(CONNECTION_STATES.CONNECTED);

            // Frame generation loop (~15 FPS)
            intervalRef.current = setInterval(() => {
                const nextFrame = generateNextFrame();
                setFrame(nextFrame);
                lastValidFrameRef.current = nextFrame;
                frameCountRef.current += 1;
            }, 66);
        }, 500); // Simulate 500ms connection time

        // FPS calculation loop
        fpsIntervalRef.current = setInterval(() => {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
        }, 1000);

        return () => {
            clearTimeout(startupTimeout);
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current);
            setConnectionStatus(CONNECTION_STATES.DISCONNECTED);
        };
    }, []);

    // Manual reconnect (not used in fake stream, but API compatible)
    const manualReconnect = () => {
        console.log('[FakeStream] Manual reconnect - no-op in fake mode');
    };

    return {
        frame,
        fps,
        connectionStatus,
        reconnectAttempt,
        lastValidFrame: lastValidFrameRef.current,
        manualReconnect
    };
};

export default useFakeStream;
