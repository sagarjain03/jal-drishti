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

        // TASK 5: Debug logging - log payload from backend
        if (detections.length > 0) {
            console.log("[DetectionOverlay] Detections from backend:", detections);
        }

        ctx.save();

        // TASK 3: Use track_id for keying (Map for deduplication)
        const detectionMap = new Map();
        detections.forEach(det => {
            const key = det.track_id ?? det.bbox.join(',');  // Use track_id if available
            detectionMap.set(key, det);
        });

        // Render each detection
        detectionMap.forEach((det, key) => {
            const { bbox, label, confidence, type, track_id } = det;

            // TASK 2: NEVER filter TACTICAL_THREAT regardless of confidence
            // TACTICAL_THREAT must ALWAYS be rendered
            const isTactical = type === 'TACTICAL_THREAT';
            const isThreat = type === 'THREAT' || isTactical;
            const isAnomaly = type === 'ANOMALY';

            // Support both [x, y, w, h] and [x1, y1, x2, y2] formats
            let x, y, w, h;
            if (bbox.length === 4) {
                // If w/h are larger than x/y, assume [x1, y1, x2, y2]
                if (bbox[2] > bbox[0] * 2) {
                    x = bbox[0];
                    y = bbox[1];
                    w = bbox[2] - bbox[0];
                    h = bbox[3] - bbox[1];
                } else {
                    [x, y, w, h] = bbox;
                }
            }

            // TASK 1: Determine color based on DETECTION TYPE, not just system state
            // TACTICAL_THREAT = always RED
            // THREAT = RED
            // ANOMALY = YELLOW
            let color, lineWidth, glowBlur;

            if (isTactical) {
                // TACTICAL OVERRIDE - Always RED, thicker, more glow
                color = '#FF0000';  // Pure red
                lineWidth = 4;
                glowBlur = 20;
            } else if (isThreat) {
                // Regular threat - Red
                color = STATE_COLORS.CONFIRMED_THREAT || '#EF4444';
                lineWidth = 3;
                glowBlur = 15;
            } else if (isAnomaly) {
                // Anomaly - Yellow
                color = STATE_COLORS.POTENTIAL_ANOMALY || '#F97316';
                lineWidth = 3;
                glowBlur = 12;
            } else {
                // Safe mode / neutral - Gray
                color = STATE_COLORS.SAFE_MODE || '#888888';
                lineWidth = 1;
                glowBlur = 0;
            }

            // Draw glow effect for threats and tactical
            if (isThreat || isTactical || isAnomaly) {
                ctx.save();
                ctx.shadowColor = color;
                ctx.shadowBlur = glowBlur;
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.strokeRect(x, y, w, h);
                ctx.restore();
            }

            // Draw main bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(x, y, w, h);

            // Corner accents for threats and tactical
            if (isThreat || isTactical) {
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

            // TASK 1: Trust backend label EXACTLY for TACTICAL_THREAT
            // Do NOT recompute label, use exactly what backend sends
            let displayLabel;
            if (isTactical) {
                // Backend already sends "LABEL [TACTICAL]" format
                // Trust it exactly - DO NOT modify
                displayLabel = label;
            } else {
                displayLabel = label || 'UNKNOWN';
            }

            const confidenceText = `${(confidence * 100).toFixed(0)}%`;

            // Draw label background with better styling
            ctx.font = 'bold 13px Inter, sans-serif';
            const labelText = displayLabel;
            const textWidth = ctx.measureText(labelText).width;
            const labelHeight = 28;
            const labelY = y - labelHeight - 4;

            // Background color matches detection type
            ctx.fillStyle = color;

            // Draw label background
            const bgX = x;
            const bgY = labelY;
            const bgWidth = textWidth + 60; // Extra space for confidence bar
            const bgHeight = labelHeight;

            ctx.save();
            ctx.shadowColor = color;
            ctx.shadowBlur = 8;
            ctx.fillRect(bgX, bgY, bgWidth, bgHeight);
            ctx.restore();

            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, bgX + 6, bgY + 17);

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
            ctx.fillStyle = '#ffffff';
            ctx.fillText(confidenceText, barX + barWidth + 4, barY + 4);

            // TACTICAL marker - add extra visual indicator
            if (isTactical) {
                ctx.font = 'bold 10px JetBrains Mono, monospace';
                ctx.fillStyle = '#00FF00';  // Green "operator" indicator
                ctx.fillText('👤 OPERATOR', bgX + bgWidth + 8, bgY + 17);
            }
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
