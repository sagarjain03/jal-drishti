import React, { useRef, useEffect, useState } from 'react';
import { STATE_COLORS, SYSTEM_STATES } from '../constants';

/**
 * DetectionOverlay Component (Enhanced)
 * 
 * Draws bounding boxes on canvas with:
 * - Color based on SYSTEM STATE (not individual confidence)
 * - SAFE_MODE: thin stroke, low opacity, "Unreliable" label
 * - Standard: 3px stroke, full opacity with GLOW EFFECTS
 * - Pulse animation on new detections
 * - Confidence shown as colored bar (Green → Red gradient)
 * 
 * IMPORTANT: Performance-optimized CSS-only animations
 */
const DetectionOverlay = ({
    detections = [],
    systemState = SYSTEM_STATES.SAFE_MODE,
    width = 640,
    height = 480
}) => {
    const canvasRef = useRef(null);
    const [pulseActive, setPulseActive] = useState(false);
    const prevDetectionCountRef = useRef(0);

    // Trigger pulse animation when new detections appear
    useEffect(() => {
        if (detections.length > prevDetectionCountRef.current) {
            setPulseActive(true);
            const timer = setTimeout(() => setPulseActive(false), 300);
            return () => clearTimeout(timer);
        }
        prevDetectionCountRef.current = detections.length;
    }, [detections.length]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Clear previous frame
        ctx.clearRect(0, 0, width, height);

        // Determine styling based on system state
        const isSafeMode = systemState === SYSTEM_STATES.SAFE_MODE;
        const isConfirmedThreat = systemState === SYSTEM_STATES.CONFIRMED_THREAT;
        const color = STATE_COLORS[systemState] || STATE_COLORS.SAFE_MODE;
        const lineWidth = isSafeMode ? 1 : (isConfirmedThreat ? 4 : 3);
        const globalAlpha = isSafeMode ? 0.5 : 1.0;

        ctx.save();
        ctx.globalAlpha = globalAlpha;

        detections.forEach((det, index) => {
            const { bbox, label, confidence } = det;

            // Support both [x, y, w, h] and [x1, y1, x2, y2] formats
            let x, y, w, h;
            if (bbox.length === 4) {
                [x, y, w, h] = bbox;
            }

            // Draw glow effect for non-safe mode
            if (!isSafeMode) {
                ctx.save();
                ctx.shadowColor = color;
                ctx.shadowBlur = isConfirmedThreat ? 15 : 10;
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.strokeRect(x, y, w, h);
                ctx.restore();
            }

            // Draw main bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(x, y, w, h);

            // Corner accents for enhanced look
            if (!isSafeMode) {
                const cornerLength = 12;
                ctx.lineWidth = lineWidth + 1;

                // Top-left corner
                ctx.beginPath();
                ctx.moveTo(x, y + cornerLength);
                ctx.lineTo(x, y);
                ctx.lineTo(x + cornerLength, y);
                ctx.stroke();

                // Top-right corner
                ctx.beginPath();
                ctx.moveTo(x + w - cornerLength, y);
                ctx.lineTo(x + w, y);
                ctx.lineTo(x + w, y + cornerLength);
                ctx.stroke();

                // Bottom-left corner
                ctx.beginPath();
                ctx.moveTo(x, y + h - cornerLength);
                ctx.lineTo(x, y + h);
                ctx.lineTo(x + cornerLength, y + h);
                ctx.stroke();

                // Bottom-right corner
                ctx.beginPath();
                ctx.moveTo(x + w - cornerLength, y + h);
                ctx.lineTo(x + w, y + h);
                ctx.lineTo(x + w, y + h - cornerLength);
                ctx.stroke();
            }

            // Prepare label text
            let displayLabel = label || 'anomaly';
            if (isSafeMode) {
                displayLabel += ' (Unreliable)';
            }
            const confidenceText = `${(confidence * 100).toFixed(0)}%`;

            // Draw label background with better styling
            ctx.font = isSafeMode ? 'bold 11px Inter, sans-serif' : 'bold 13px Inter, sans-serif';
            const labelText = `${displayLabel}`;
            const textWidth = ctx.measureText(labelText).width;
            const labelHeight = isSafeMode ? 22 : 28;
            const labelY = y - labelHeight - 4;

            // Background with rounded corners effect
            ctx.fillStyle = isSafeMode ? 'rgba(0, 0, 0, 0.6)' : color;

            // Draw label background
            const bgX = x;
            const bgY = labelY;
            const bgWidth = textWidth + 60; // Extra space for confidence bar
            const bgHeight = labelHeight;

            ctx.save();
            if (!isSafeMode) {
                ctx.shadowColor = color;
                ctx.shadowBlur = 8;
            }
            ctx.fillRect(bgX, bgY, bgWidth, bgHeight);
            ctx.restore();

            // Draw label text
            ctx.fillStyle = isSafeMode ? '#ffffff' : (isConfirmedThreat ? '#ffffff' : '#000000');
            ctx.fillText(labelText, bgX + 6, bgY + (isSafeMode ? 14 : 17));

            // Draw confidence bar (Green → Red gradient based on confidence)
            const barWidth = 40;
            const barHeight = 4;
            const barX = bgX + textWidth + 12;
            const barY = bgY + (labelHeight / 2) - 2;

            // Background bar
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.fillRect(barX, barY, barWidth, barHeight);

            // Filled bar with gradient color based on confidence
            const fillWidth = barWidth * confidence;
            const confidenceColor = getConfidenceColor(confidence);
            ctx.fillStyle = confidenceColor;
            ctx.fillRect(barX, barY, fillWidth, barHeight);

            // Confidence percentage text
            ctx.font = 'bold 10px JetBrains Mono, monospace';
            ctx.fillStyle = isSafeMode ? '#aaaaaa' : '#ffffff';
            ctx.fillText(confidenceText, barX + barWidth + 4, barY + 4);
        });

        ctx.restore();

    }, [detections, systemState, width, height]);

    /**
     * Get color based on confidence level
     * Low confidence → Green, High confidence → Red
     */
    function getConfidenceColor(confidence) {
        if (confidence < 0.4) {
            return '#22C55E'; // Green
        } else if (confidence < 0.7) {
            return '#F97316'; // Amber
        } else {
            return '#EF4444'; // Red
        }
    }

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className={`detection-overlay ${pulseActive ? 'detection-pulse' : ''}`}
            style={{
                transition: pulseActive ? 'filter 0.3s ease' : 'none',
                filter: pulseActive ? 'brightness(1.2)' : 'none'
            }}
        />
    );
};

export default DetectionOverlay;
