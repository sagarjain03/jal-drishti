import React from 'react';

/**
 * SonarTemporalGraph (v3 — Props-Driven)
 * 
 * Accepts time-series data via props instead of generating internally.
 * Shows: Distance, Signal Strength, Noise Level, SNR.
 */
const SonarTemporalGraph = ({ timeSeries = {} }) => {
    const { signal = [], noise = [], distance = [], snr = [] } = timeSeries;

    // Use distance graph as primary, show others as secondary
    const distancePoints = distance.slice(-20);
    const signalPoints = signal.slice(-20);
    const noisePoints = noise.slice(-20);

    const svgWidth = 500;
    const svgHeight = 200;
    const padding = { top: 20, bottom: 30, left: 45, right: 15 };
    const graphWidth = svgWidth - padding.left - padding.right;
    const graphHeight = svgHeight - padding.top - padding.bottom;

    // Scale distance (20-500m) → graph Y
    const scaleY = (value, min, max) => {
        const normalized = (value - min) / (max - min + 0.001);
        return padding.top + graphHeight * (1 - normalized);
    };

    // Build path from data
    const buildPath = (data, valueKey, min, max) => {
        if (data.length < 2) return '';
        const xStep = graphWidth / (data.length - 1);
        return data.map((d, i) => {
            const x = padding.left + i * xStep;
            const y = scaleY(d.value, min, max);
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
    };

    // Compute mins/maxes
    const distMin = distancePoints.length > 0 ? Math.min(...distancePoints.map(d => d.value)) - 20 : 0;
    const distMax = distancePoints.length > 0 ? Math.max(...distancePoints.map(d => d.value)) + 20 : 500;

    const distancePath = buildPath(distancePoints, 'value', distMin, distMax);
    const signalPath = buildPath(signalPoints, 'value', 0, 1);
    const noisePath = buildPath(noisePoints, 'value', 0, 1);

    // Y-axis labels for distance
    const yLabels = [distMax, (distMax + distMin) / 2, distMin].map(v => Math.round(v));

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <svg width="100%" height="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="xMidYMid meet">
                {/* Background grid */}
                {[0.25, 0.5, 0.75].map(frac => (
                    <line
                        key={frac}
                        x1={padding.left}
                        y1={padding.top + graphHeight * frac}
                        x2={svgWidth - padding.right}
                        y2={padding.top + graphHeight * frac}
                        stroke="#27272A" // Zinc-800
                        strokeDasharray="4,4"
                    />
                ))}

                {/* Y-axis labels */}
                {yLabels.map((label, i) => (
                    <text
                        key={label}
                        x={padding.left - 5}
                        y={padding.top + (i * graphHeight / 2) + 4}
                        textAnchor="end"
                        fill="#71717A" // Zinc-500
                        fontSize="9"
                        fontFamily="'JetBrains Mono', monospace"
                    >
                        {label}m
                    </text>
                ))}

                {/* Distance line */}
                {distancePath && (
                    <path d={distancePath} fill="none" stroke="#60A5FA" strokeWidth="2" strokeLinejoin="round" />
                )}

                {/* Signal line (scaled to graph) */}
                {signalPath && (
                    <path d={signalPath} fill="none" stroke="#22C55E" strokeWidth="1.5" strokeLinejoin="round" opacity="0.6" />
                )}

                {/* Noise line (scaled to graph) */}
                {noisePath && (
                    <path d={noisePath} fill="none" stroke="#EF4444" strokeWidth="1.5" strokeLinejoin="round" opacity="0.6" />
                )}

                {/* Current value markers */}
                {distancePoints.length > 0 && (() => {
                    const last = distancePoints[distancePoints.length - 1];
                    const x = padding.left + (distancePoints.length - 1) * (graphWidth / (distancePoints.length - 1));
                    const y = scaleY(last.value, distMin, distMax);
                    return (
                        <>
                            <circle cx={x} cy={y} r="4" fill="#60A5FA" stroke="#18181B" strokeWidth="2" />
                            <text x={x + 8} y={y + 4} fill="#60A5FA" fontSize="10" fontFamily="'JetBrains Mono', monospace" fontWeight="700">
                                {Math.round(last.value)}m
                            </text>
                        </>
                    );
                })()}

                {/* X-axis label */}
                <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fill="#52525B" fontSize="9" fontFamily="'JetBrains Mono', monospace">
                    TIME (30s window)
                </text>
            </svg>

            {/* Legend */}
            <div style={{ position: 'absolute', top: '5px', right: '10px', display: 'flex', gap: '10px', fontSize: '9px' }}>
                <span style={{ color: '#3B82F6' }}>● Distance</span>
                <span style={{ color: '#22C55E' }}>● Signal</span>
                <span style={{ color: '#EF4444' }}>● Noise</span>
            </div>

            {distancePoints.length === 0 && (
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#475569', fontSize: '12px' }}>
                    AWAITING SONAR DATA...
                </div>
            )}
        </div>
    );
};

export default SonarTemporalGraph;
