import React, { useState, useEffect } from 'react';

/**
 * SonarRadar Component
 * Circular radar with concentric zones and rotating sweep
 */
const SonarRadar = ({ detections = [] }) => {
    const [sweepAngle, setSweepAngle] = useState(0);
    const centerX = 250;
    const centerY = 250;
    const maxRadius = 220;

    // Zones (in meters)
    const zones = [
        { range: 50, radius: maxRadius * 0.3, color: '#EF4444', label: 'DANGER', opacity: 0.1 },
        { range: 150, radius: maxRadius * 0.6, color: '#F97316', label: 'WARNING', opacity: 0.08 },
        { range: 500, radius: maxRadius, color: '#22C55E', label: 'DETECTION', opacity: 0.05 }
    ];

    // Rotate sweep
    useEffect(() => {
        const interval = setInterval(() => {
            setSweepAngle(prev => (prev + 1) % 360);
        }, 50);
        return () => clearInterval(interval);
    }, []);

    // Convert distance to radius
    const distanceToRadius = (distance) => {
        return (distance / 500) * maxRadius;
    };

    // Get detection color based on confidence
    const getDetectionColor = (confidence) => {
        if (confidence > 0.8) return '#EF4444'; // Red
        if (confidence > 0.6) return '#F97316'; // Amber
        return '#22C55E'; // Green
    };

    return (
        <div style={{
            position: 'relative',
            width: '100%',
            height: '100%',
            background: '#09090B', // Matte black
            borderRadius: '8px',
            border: '1px solid #27272A', // Zinc-800
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden'
        }}>
            <svg viewBox={`0 0 ${2 * centerX} ${2 * centerY}`} style={{ width: '100%', height: '100%', maxHeight: '100%', maxWidth: '100%' }} preserveAspectRatio="xMidYMid meet">
                {/* Concentric Zones */}
                {zones.map((zone, idx) => (
                    <g key={idx}>
                        <circle
                            cx={centerX}
                            cy={centerY}
                            r={zone.radius}
                            fill={zone.color}
                            fillOpacity={zone.opacity}
                            stroke={zone.color}
                            strokeWidth="1"
                            strokeOpacity="0.2"
                        />
                        {/* Zone Label */}
                        <text
                            x={centerX}
                            y={centerY - zone.radius + 12}
                            fill={zone.color}
                            fontSize="9"
                            fontWeight="600"
                            textAnchor="middle"
                            opacity="0.8"
                            style={{ pointerEvents: 'none' }}
                        >
                            {zone.range}m
                        </text>
                    </g>
                ))}

                {/* Grid Lines */}
                {[0, 45, 90, 135, 180, 225, 270, 315].map(angle => {
                    const rad = (angle * Math.PI) / 180;
                    const x2 = centerX + maxRadius * Math.cos(rad);
                    const y2 = centerY + maxRadius * Math.sin(rad);
                    return (
                        <line
                            key={angle}
                            x1={centerX}
                            y1={centerY}
                            x2={x2}
                            y2={y2}
                            stroke="#3F3F46" // Zinc-700
                            strokeWidth="1"
                            opacity="0.4"
                        />
                    );
                })}

                {/* Rotating Sweep */}
                <defs>
                    <linearGradient id="sweepGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#22C55E" stopOpacity="0" />
                        <stop offset="100%" stopColor="#22C55E" stopOpacity="0.15" />
                    </linearGradient>
                </defs>
                <path
                    d={`M ${centerX} ${centerY} L ${centerX + maxRadius * Math.cos((sweepAngle * Math.PI) / 180)} ${centerY + maxRadius * Math.sin((sweepAngle * Math.PI) / 180)} A ${maxRadius} ${maxRadius} 0 0 1 ${centerX + maxRadius * Math.cos(((sweepAngle - 60) * Math.PI) / 180)} ${centerY + maxRadius * Math.sin(((sweepAngle - 60) * Math.PI) / 180)} Z`}
                    fill="url(#sweepGradient)"
                    opacity="0.8"
                />

                {/* Detection Points */}
                {detections.map((detection, idx) => {
                    const radius = distanceToRadius(detection.distance);
                    const angle = detection.angle || (idx * 120); // Spread if no angle
                    const rad = (angle * Math.PI) / 180;
                    const x = centerX + radius * Math.cos(rad);
                    const y = centerY + radius * Math.sin(rad);
                    const color = getDetectionColor(detection.confidence);

                    return (
                        <g key={idx}>
                            {/* Glow */}
                            <circle
                                cx={x}
                                cy={y}
                                r="12"
                                fill={color}
                                opacity="0.15"
                            >
                                <animate
                                    attributeName="r"
                                    values="12;18;12"
                                    dur="2s"
                                    repeatCount="indefinite"
                                />
                            </circle>
                            {/* Dot */}
                            <circle
                                cx={x}
                                cy={y}
                                r="4"
                                fill={color}
                                stroke="#09090B"
                                strokeWidth="1.5"
                                opacity="1"
                            />
                            {/* Label */}
                            <text
                                x={x}
                                y={y - 15}
                                fill="#E4E4E7" // Zinc-200
                                fontSize="10"
                                fontWeight="600"
                                textAnchor="middle"
                                style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}
                            >
                                {detection.distance}m
                            </text>
                            <text
                                x={x}
                                y={y + 25}
                                fill={color}
                                fontSize="9"
                                fontWeight="700"
                                textAnchor="middle"
                                style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}
                            >
                                {(detection.confidence * 100).toFixed(0)}%
                            </text>
                        </g>
                    );
                })}

                {/* Center Marker */}
                <circle
                    cx={centerX}
                    cy={centerY}
                    r="5"
                    fill="#22C55E"
                    opacity="0.8"
                />
                <circle
                    cx={centerX}
                    cy={centerY}
                    r="2"
                    fill="#09090B"
                />
            </svg>

            {/* Compass Labels */}
            <div style={{ position: 'absolute', top: '10px', left: '50%', transform: 'translateX(-50%)', color: '#52525B', fontSize: '10px', fontWeight: '700' }}>N</div>
            <div style={{ position: 'absolute', bottom: '10px', left: '50%', transform: 'translateX(-50%)', color: '#52525B', fontSize: '10px', fontWeight: '700' }}>S</div>
            <div style={{ position: 'absolute', top: '50%', left: '10px', transform: 'translateY(-50%)', color: '#52525B', fontSize: '10px', fontWeight: '700' }}>W</div>
            <div style={{ position: 'absolute', top: '50%', right: '10px', transform: 'translateY(-50%)', color: '#52525B', fontSize: '10px', fontWeight: '700' }}>E</div>
        </div>
    );
};

export default SonarRadar;
