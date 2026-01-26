/**
 * mockStreamGenerator.js
 * 
 * Updated for Phase-2 testing.
 * Supports all 3 system states: SAFE_MODE, POTENTIAL_ANOMALY, CONFIRMED_THREAT
 * State cycles automatically for TEST 4 validation.
 */

import { SYSTEM_STATES } from '../constants';

// Simulation constants
const WIDTH = 640;
const HEIGHT = 480;
const MAX_OBJECTS = 3;

// State to persist between frames
let trackedObjects = [
    {
        id: 1,
        label: 'anomaly',
        x: 120,
        y: 80,
        w: 150,
        h: 120,
        vx: 2,
        vy: 1.5,
        confidence: 0.65
    }
];

let frameId = 0;
let currentStateIndex = 0;
let framesInState = 0;
const FRAMES_PER_STATE = 150; // ~10 seconds per state at 15 FPS

// State cycle for TEST 4
const STATE_CYCLE = [
    SYSTEM_STATES.SAFE_MODE,
    SYSTEM_STATES.POTENTIAL_ANOMALY,
    SYSTEM_STATES.CONFIRMED_THREAT,
    SYSTEM_STATES.SAFE_MODE
];

// Force a specific state for manual testing (set to null for auto-cycle)
let forcedState = null;

export const setForcedState = (state) => {
    forcedState = state;
};

export const clearForcedState = () => {
    forcedState = null;
};

export const generateNextFrame = () => {
    frameId++;
    framesInState++;

    // Auto-cycle states for TEST 4 (if not forced)
    if (!forcedState && framesInState >= FRAMES_PER_STATE) {
        framesInState = 0;
        currentStateIndex = (currentStateIndex + 1) % STATE_CYCLE.length;
    }

    const currentState = forcedState || STATE_CYCLE[currentStateIndex];

    // Update tracked objects
    trackedObjects.forEach(obj => {
        // Move
        obj.x += obj.vx;
        obj.y += obj.vy;

        // Bounce off walls
        if (obj.x <= 0 || obj.x + obj.w >= WIDTH) obj.vx *= -1;
        if (obj.y <= 0 || obj.y + obj.h >= HEIGHT) obj.vy *= -1;

        // Adjust confidence based on state
        if (currentState === SYSTEM_STATES.CONFIRMED_THREAT) {
            obj.confidence = 0.85 + Math.random() * 0.1; // 85-95%
        } else if (currentState === SYSTEM_STATES.POTENTIAL_ANOMALY) {
            obj.confidence = 0.55 + Math.random() * 0.15; // 55-70%
        } else {
            obj.confidence = 0.25 + Math.random() * 0.1; // 25-35% (SAFE_MODE)
        }
    });

    // Add new object occasionally
    if (trackedObjects.length < MAX_OBJECTS && Math.random() > 0.995) {
        trackedObjects.push({
            id: Date.now(),
            label: 'anomaly',
            x: Math.random() * (WIDTH - 100),
            y: Math.random() * (HEIGHT - 100),
            w: 80 + Math.random() * 50,
            h: 80 + Math.random() * 50,
            vx: (Math.random() - 0.5) * 3,
            vy: (Math.random() - 0.5) * 3,
            confidence: 0.5
        });
    }

    // Remove object occasionally
    if (trackedObjects.length > 1 && Math.random() > 0.998) {
        trackedObjects.shift();
    }

    // Calculate max confidence
    const maxConfidence = trackedObjects.length > 0
        ? Math.max(...trackedObjects.map(obj => obj.confidence))
        : 0;

    // Construct detections
    const detections = trackedObjects.map(obj => ({
        label: obj.label,
        confidence: parseFloat(obj.confidence.toFixed(2)),
        bbox: [Math.floor(obj.x), Math.floor(obj.y), Math.floor(obj.w), Math.floor(obj.h)]
    }));

    // Simulate system metrics
    const simulatedLatency = 50 + Math.floor(Math.random() * 30);
    const simulatedMlFps = 14 + Math.random() * 2;

    return {
        status: 'success',
        frame_id: frameId,
        timestamp: new Date().toISOString(),
        state: currentState,
        max_confidence: parseFloat(maxConfidence.toFixed(2)),
        image_data: `https://placehold.co/640x480/1a1a2e/FFF?text=Frame+${frameId}`,
        detections: detections,
        visibility_score: currentState === SYSTEM_STATES.SAFE_MODE ? 0.3 : 0.8,
        system: {
            fps: parseFloat(simulatedMlFps.toFixed(1)),
            latency_ms: simulatedLatency
        }
    };
};

// Reset function for testing
export const resetMockStream = () => {
    frameId = 0;
    currentStateIndex = 0;
    framesInState = 0;
    trackedObjects = [{
        id: 1,
        label: 'anomaly',
        x: 120,
        y: 80,
        w: 150,
        h: 120,
        vx: 2,
        vy: 1.5,
        confidence: 0.65
    }];
};
