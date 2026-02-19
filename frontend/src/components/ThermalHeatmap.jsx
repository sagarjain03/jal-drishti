import React from 'react';

/**
 * ThermalHeatmap (v3 — Props-Driven)
 * 
 * Accepts heatmap grid data via props instead of generating internally.
 * Renders a color-coded grid with temperature scale.
 */
const ThermalHeatmap = ({ heatData = [] }) => {
    const gridSize = heatData.length > 0 ? Math.round(Math.sqrt(heatData.length)) : 12;
    const cellSize = 30;
    const svgSize = gridSize * cellSize;

    // Temperature color mapping
    const intensityToColor = (intensity) => {
        if (intensity > 0.7) return `rgba(239, 68, 68, ${0.5 + intensity * 0.5})`;   // Red - hot
        if (intensity > 0.4) return `rgba(249, 115, 22, ${0.3 + intensity * 0.5})`;  // Orange - warm
        if (intensity > 0.15) return `rgba(59, 130, 246, ${0.2 + intensity * 0.5})`; // Blue - mild
        return `rgba(30, 41, 59, ${0.3 + intensity * 0.3})`;                         // Dark - cold
    };

    return (
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', justifyContent: 'center' }}>
            <svg width={svgSize} height={svgSize} style={{ borderRadius: '6px', overflow: 'hidden' }}>
                {heatData.map((cell, idx) => (
                    <rect
                        key={idx}
                        x={cell.x * cellSize}
                        y={cell.y * cellSize}
                        width={cellSize - 1}
                        height={cellSize - 1}
                        fill={intensityToColor(cell.intensity)}
                        rx="2"
                    />
                ))}

                {/* Grid overlay */}
                {Array.from({ length: gridSize + 1 }).map((_, i) => (
                    <React.Fragment key={i}>
                        <line x1={i * cellSize} y1={0} x2={i * cellSize} y2={svgSize} stroke="rgba(100,140,255,0.05)" />
                        <line x1={0} y1={i * cellSize} x2={svgSize} y2={i * cellSize} stroke="rgba(100,140,255,0.05)" />
                    </React.Fragment>
                ))}
            </svg>

            {/* Temperature Scale */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                <span style={{ color: '#64748B', fontSize: '8px', marginBottom: '4px' }}>°C</span>
                <div style={{
                    width: '12px',
                    height: '120px',
                    borderRadius: '4px',
                    background: 'linear-gradient(180deg, #EF4444, #F97316, #3B82F6, #1E293B)',
                    border: '1px solid rgba(100,140,255,0.1)'
                }} />
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '120px', fontSize: '8px', color: '#64748B' }}>
                    <span>HIGH</span>
                    <span>MED</span>
                    <span>LOW</span>
                </div>
            </div>

            {heatData.length === 0 && (
                <div style={{ position: 'absolute', color: '#475569', fontSize: '12px' }}>
                    AWAITING IR DATA...
                </div>
            )}
        </div>
    );
};

export default ThermalHeatmap;
