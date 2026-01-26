import React, { useRef, useEffect } from 'react';
import { STATE_COLORS, SYSTEM_STATES } from '../constants';

/**
 * DetectionOverlay Component
 * 
 * Draws bounding boxes on canvas with:
 * - Color based on SYSTEM STATE (not individual confidence)
 * - SAFE_MODE: thin stroke, low opacity, "Unreliable" label
 * - Standard: 3px stroke, full opacity
 */
const DetectionOverlay = ({
    detections = [],
    systemState = SYSTEM_STATES.SAFE_MODE,
    width = 640,
    height = 480
}) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Clear previous frame
        ctx.clearRect(0, 0, width, height);

        // Determine styling based on system state
        const isSafeMode = systemState === SYSTEM_STATES.SAFE_MODE;
        const color = STATE_COLORS[systemState] || STATE_COLORS.SAFE_MODE;
        const lineWidth = isSafeMode ? 1 : 3;
        const globalAlpha = isSafeMode ? 0.4 : 1.0;

        ctx.save();
        ctx.globalAlpha = globalAlpha;

        detections.forEach((det) => {
            const { bbox, label, confidence } = det;

            // Support both [x, y, w, h] and [x1, y1, x2, y2] formats
            let x, y, w, h;
            if (bbox.length === 4) {
                // Assume [x, y, w, h] format based on existing code
                [x, y, w, h] = bbox;
            }

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(x, y, w, h);

            // Prepare label text
            let displayLabel = label || 'anomaly';
            if (isSafeMode) {
                displayLabel += ' (Unreliable)';
            }
            const confidenceText = `${(confidence * 100).toFixed(0)}%`;
            const text = `${displayLabel} ${confidenceText}`;

            // Draw label background
            ctx.font = isSafeMode ? '12px sans-serif' : '14px sans-serif';
            const textWidth = ctx.measureText(text).width;
            const labelHeight = isSafeMode ? 20 : 25;

            ctx.fillStyle = color;
            ctx.fillRect(x, y - labelHeight, textWidth + 10, labelHeight);

            // Draw text
            ctx.fillStyle = isSafeMode ? '#ffffff' : '#000000';
            ctx.fillText(text, x + 5, y - (isSafeMode ? 5 : 7));
        });

        ctx.restore();

    }, [detections, systemState, width, height]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="detection-overlay"
        />
    );
};

export default DetectionOverlay;
