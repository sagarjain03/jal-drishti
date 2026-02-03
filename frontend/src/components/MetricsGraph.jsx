import React, { useMemo } from 'react';

/**
 * MetricsGraph Component
 * 
 * Lightweight SVG-based real-time line chart.
 * Features:
 * - Smooth polyline rendering
 * - Auto-scaling Y-axis
 * - Minimal grid lines
 * - Configurable colors and labels
 * - CSS-based smooth updates
 */
const MetricsGraph = ({
    data = [],
    width = 200,
    height = 60,
    color = '#00d4ff',
    label = 'Metric',
    unit = '',
    maxPoints = 60,
    minValue = 0,
    maxValue = null
}) => {
    // Calculate graph dimensions with padding
    const padding = { top: 8, right: 8, bottom: 18, left: 8 };
    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    // Calculate Y-axis bounds
    const yBounds = useMemo(() => {
        if (data.length === 0) return { min: minValue, max: 100 };

        const dataMax = Math.max(...data);
        const dataMin = Math.min(...data, minValue);

        return {
            min: minValue,
            max: maxValue !== null ? maxValue : Math.max(dataMax * 1.1, 10)
        };
    }, [data, minValue, maxValue]);

    // Generate SVG path from data
    const pathData = useMemo(() => {
        if (data.length < 2) return '';

        const points = data.slice(-maxPoints).map((value, index, arr) => {
            const x = padding.left + (index / (arr.length - 1)) * graphWidth;
            const normalizedValue = (value - yBounds.min) / (yBounds.max - yBounds.min);
            const y = padding.top + graphHeight - (normalizedValue * graphHeight);
            return `${x},${y}`;
        });

        return `M${points.join(' L')}`;
    }, [data, maxPoints, graphWidth, graphHeight, yBounds, padding]);

    // Generate area fill path
    const areaPath = useMemo(() => {
        if (data.length < 2) return '';

        const points = data.slice(-maxPoints).map((value, index, arr) => {
            const x = padding.left + (index / (arr.length - 1)) * graphWidth;
            const normalizedValue = (value - yBounds.min) / (yBounds.max - yBounds.min);
            const y = padding.top + graphHeight - (normalizedValue * graphHeight);
            return `${x},${y}`;
        });

        const startX = padding.left;
        const endX = padding.left + graphWidth;
        const bottomY = padding.top + graphHeight;

        return `M${startX},${bottomY} L${points.join(' L')} L${endX},${bottomY} Z`;
    }, [data, maxPoints, graphWidth, graphHeight, yBounds, padding]);

    // Current value (latest)
    const currentValue = data.length > 0 ? data[data.length - 1] : 0;

    return (
        <div className="metrics-graph" style={{ width, height }}>
            <svg width={width} height={height}>
                {/* Background */}
                <rect
                    x={padding.left}
                    y={padding.top}
                    width={graphWidth}
                    height={graphHeight}
                    fill="rgba(0, 0, 0, 0.3)"
                    rx="4"
                />

                {/* Horizontal grid lines (2 lines) */}
                <line
                    x1={padding.left}
                    y1={padding.top + graphHeight / 2}
                    x2={padding.left + graphWidth}
                    y2={padding.top + graphHeight / 2}
                    stroke="rgba(255, 255, 255, 0.1)"
                    strokeWidth="1"
                    strokeDasharray="4,4"
                />

                {/* Area fill under the line */}
                {areaPath && (
                    <path
                        d={areaPath}
                        fill={`${color}20`}
                    />
                )}

                {/* Main line */}
                {pathData && (
                    <path
                        d={pathData}
                        fill="none"
                        stroke={color}
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{
                            filter: `drop-shadow(0 0 4px ${color}60)`
                        }}
                    />
                )}

                {/* Current value dot */}
                {data.length > 0 && (
                    <circle
                        cx={padding.left + graphWidth}
                        cy={padding.top + graphHeight - ((currentValue - yBounds.min) / (yBounds.max - yBounds.min)) * graphHeight}
                        r="3"
                        fill={color}
                        style={{
                            filter: `drop-shadow(0 0 6px ${color})`
                        }}
                    />
                )}
            </svg>

            {/* Label and current value */}
            <div className="metrics-graph-label">
                <span className="metrics-graph-name">{label}</span>
                <span className="metrics-graph-value" style={{ color }}>
                    {currentValue.toFixed(unit === 'ms' ? 0 : 1)}{unit}
                </span>
            </div>
        </div>
    );
};

export default MetricsGraph;
