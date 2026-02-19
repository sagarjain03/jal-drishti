import React, { useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import ThermalHeatmap from '../components/ThermalHeatmap';
import { useInfraredEngine } from '../sensors/infrared/infraredEngine';
import { useSystemState } from '../context/SystemStateContext';

/**
 * Infrared Page (v3 — Engine-Driven)
 * 
 * All data from useInfraredEngine() — zero hardcoded values.
 * Panels: Zone Analysis, Stability, Material Signature, Drift Rate,
 *         Area, Confidence Trend, Correlation Score
 */
const Infrared = () => {
    const { systemStatus } = useOutletContext();
    const { updateIRData, correlation } = useSystemState();

    // IR engine: feeds SystemStateManager via callback
    const ir = useInfraredEngine(useCallback((data) => {
        updateIRData(data);
    }, [updateIRData]));

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '15px', gap: '12px', overflowY: 'auto', background: '#0A0A0F' }}>
            {/* PAGE HEADER */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ margin: 0, color: '#FF8B8B', fontSize: '16px', letterSpacing: '2px', fontFamily: '"JetBrains Mono", monospace' }}>
                    INFRARED ANALYSIS
                </h2>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <StatusBadge label="STABILITY" value={ir.thermalStability} quality={ir.thermalStability === 'CONSISTENT' ? 'GOOD' : ir.thermalStability === 'MODERATE' ? 'MODERATE' : 'POOR'} />
                    <StatusBadge label="HEALTH" value={`${Math.round(ir.sensorHealth * 100)}%`} quality={ir.sensorHealth > 0.7 ? 'GOOD' : ir.sensorHealth > 0.4 ? 'MODERATE' : 'POOR'} />
                    <StatusBadge label="SIGNATURE" value={ir.signatureType} quality={ir.signatureType === 'METALLIC' ? 'POOR' : ir.signatureType === 'ORGANIC' ? 'MODERATE' : 'GOOD'} />
                </div>
            </div>

            {/* TOP ROW: Heatmap + Zone Analysis */}
            <div style={{ display: 'flex', gap: '12px', flex: 1, minHeight: '350px' }}>
                {/* Thermal Heatmap */}
                <div style={{ flex: 1, background: 'rgba(25,15,15,0.9)', borderRadius: '10px', border: '1px solid rgba(255,100,100,0.1)', padding: '12px', display: 'flex', flexDirection: 'column' }}>
                    <SectionHeader title="THERMAL HEATMAP" subtitle={`Δ${ir.heatDelta}°C | BG: ${ir.backgroundTemp}°C`} />
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <ThermalHeatmap heatData={ir.heatmapGrid} />
                    </div>
                </div>

                {/* Zone Analysis Table */}
                <div style={{ flex: 1, background: 'rgba(25,15,15,0.9)', borderRadius: '10px', border: '1px solid rgba(255,100,100,0.1)', padding: '12px', display: 'flex', flexDirection: 'column' }}>
                    <SectionHeader title="ZONE ANALYSIS" subtitle={`${ir.zoneSummaries.length} ZONES`} />
                    <div style={{ flex: 1, overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
                            <thead>
                                <tr style={{ color: '#64748B', borderBottom: '1px solid rgba(255,100,100,0.1)' }}>
                                    <th style={{ padding: '4px', textAlign: 'left' }}>ZONE</th>
                                    <th style={{ padding: '4px', textAlign: 'right' }}>TEMP</th>
                                    <th style={{ padding: '4px', textAlign: 'right' }}>Δ</th>
                                    <th style={{ padding: '4px', textAlign: 'right' }}>STABILITY</th>
                                    <th style={{ padding: '4px', textAlign: 'right' }}>DRIFT</th>
                                    <th style={{ padding: '4px', textAlign: 'right' }}>AREA</th>
                                    <th style={{ padding: '4px', textAlign: 'center' }}>LEVEL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {ir.zoneSummaries.map(zone => (
                                    <tr key={zone.id} style={{ borderBottom: '1px solid rgba(255,100,100,0.05)' }}>
                                        <td style={{ padding: '4px', color: '#CBD5E1' }}>{zone.label}</td>
                                        <td style={{ padding: '4px', textAlign: 'right', color: '#CBD5E1' }}>{zone.temp}°C</td>
                                        <td style={{ padding: '4px', textAlign: 'right', color: zone.heatDelta > 8 ? '#EF4444' : zone.heatDelta > 4 ? '#F97316' : '#22C55E' }}>
                                            +{zone.heatDelta}°C
                                        </td>
                                        <td style={{ padding: '4px', textAlign: 'right', color: zone.stability > 0.7 ? '#22C55E' : '#F97316' }}>
                                            {(zone.stability * 100).toFixed(0)}%
                                        </td>
                                        <td style={{ padding: '4px', textAlign: 'right', color: Math.abs(zone.driftRate) > 1 ? '#F97316' : '#94A3B8' }}>
                                            {zone.driftRate > 0 ? '+' : ''}{zone.driftRate}°/s
                                        </td>
                                        <td style={{ padding: '4px', textAlign: 'right', color: '#94A3B8' }}>{zone.area} m²</td>
                                        <td style={{ padding: '4px', textAlign: 'center' }}>
                                            <span style={{
                                                background: zone.level === 'HIGH' ? '#EF4444' : zone.level === 'MEDIUM' ? '#F97316' : '#22C55E',
                                                color: '#000',
                                                padding: '1px 6px',
                                                borderRadius: '3px',
                                                fontSize: '8px',
                                                fontWeight: 700
                                            }}>
                                                {zone.level}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* BOTTOM ROW: Data Panels */}
            <div style={{ display: 'flex', gap: '12px', height: '200px' }}>
                {/* Material Signature */}
                <DataPanel title="MATERIAL SIGNATURE" style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}>
                        <span style={{
                            color: ir.signatureType === 'METALLIC' ? '#EF4444' : ir.signatureType === 'ORGANIC' ? '#F97316' : '#22C55E',
                            fontSize: '14px',
                            fontWeight: 700
                        }}>
                            {ir.signatureType}
                        </span>
                    </div>
                    <CompositionBar label="Thermal Inertia" value={ir.materialBreakdown.thermalInertia} max={0.4} color="#EF4444" />
                    <CompositionBar label="Diffusion Rate" value={ir.materialBreakdown.diffusionRate} max={0.35} color="#F97316" />
                    <CompositionBar label="Shape Coherence" value={ir.materialBreakdown.shapeCoherence} max={0.25} color="#A855F7" />
                </DataPanel>

                {/* Stability Index */}
                <DataPanel title="STABILITY INDEX" style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
                        <div style={{ textAlign: 'center' }}>
                            <div style={{
                                fontSize: '32px',
                                fontWeight: 700,
                                color: ir.avgStability > 0.7 ? '#22C55E' : ir.avgStability > 0.4 ? '#F97316' : '#EF4444',
                                fontFamily: '"JetBrains Mono", monospace'
                            }}>
                                {(ir.avgStability * 100).toFixed(0)}%
                            </div>
                            <div style={{ color: '#64748B', fontSize: '10px', marginTop: '4px' }}>{ir.thermalStability}</div>
                        </div>
                    </div>
                </DataPanel>

                {/* Confidence Trend */}
                <DataPanel title="CONFIDENCE TREND" style={{ flex: 1.5 }}>
                    <ConfidenceTrendGraph data={ir.confidenceTrend} />
                </DataPanel>

                {/* Correlation Score */}
                <DataPanel title="CORRELATION" style={{ flex: 1 }}>
                    <MetricRow label="Score" value={((correlation?.correlationScore || 0) * 100).toFixed(0) + '%'} />
                    <MetricRow label="Boost" value={`+${((correlation?.correlationBoost || 0) * 100).toFixed(0)}%`} />
                    <div style={{ borderTop: '1px solid rgba(255,100,100,0.15)', marginTop: '4px', paddingTop: '4px' }}>
                        <span style={{ color: '#64748B', fontSize: '9px' }}>BREAKDOWN</span>
                        <MetricRow label="Sonar↔Cam" value={((correlation?.breakdown?.sonarCamera || 0) * 100).toFixed(0) + '%'} />
                        <MetricRow label="IR↔Sonar" value={((correlation?.breakdown?.irSonar || 0) * 100).toFixed(0) + '%'} />
                        <MetricRow label="IR↔Camera" value={((correlation?.breakdown?.irCamera || 0) * 100).toFixed(0) + '%'} />
                    </div>
                    <div style={{ marginTop: '6px', textAlign: 'center', fontSize: '9px', color: '#64748B' }}>
                        {correlation?.reason || 'No data'}
                    </div>
                </DataPanel>
            </div>
        </div>
    );
};

/* ─── Reusable Sub-Components ──────────────────────────── */

const SectionHeader = ({ title, subtitle }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ color: '#FF8B8B', fontWeight: 700, fontSize: '11px', letterSpacing: '1px' }}>{title}</span>
        <span style={{ color: '#475569', fontSize: '9px' }}>{subtitle}</span>
    </div>
);

const StatusBadge = ({ label, value, quality }) => {
    const color = quality === 'GOOD' ? '#22C55E' : quality === 'MODERATE' ? '#F97316' : '#EF4444';
    return (
        <div style={{ background: 'rgba(25,15,15,0.8)', border: `1px solid ${color}30`, borderRadius: '6px', padding: '4px 10px', display: 'flex', gap: '6px', alignItems: 'center' }}>
            <span style={{ color: '#64748B', fontSize: '9px' }}>{label}</span>
            <span style={{ color, fontSize: '11px', fontWeight: 700 }}>{value}</span>
        </div>
    );
};

const DataPanel = ({ title, children, style = {} }) => (
    <div style={{ background: 'rgba(25,15,15,0.9)', borderRadius: '10px', border: '1px solid rgba(255,100,100,0.1)', padding: '10px', display: 'flex', flexDirection: 'column', ...style }}>
        <span style={{ color: '#FF8B8B', fontWeight: 700, fontSize: '10px', letterSpacing: '0.5px', marginBottom: '8px' }}>{title}</span>
        {children}
    </div>
);

const MetricRow = ({ label, value, warn = false }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
        <span style={{ color: '#94A3B8' }}>{label}</span>
        <span style={{ color: warn ? '#F97316' : '#CBD5E1', fontWeight: 500 }}>{value}</span>
    </div>
);

const CompositionBar = ({ label, value, max, color }) => {
    const percentage = Math.min(100, (value / max) * 100);
    return (
        <div style={{ marginBottom: '5px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '2px' }}>
                <span style={{ color: '#94A3B8' }}>{label}</span>
                <span style={{ color, fontWeight: 600 }}>{value.toFixed(2)}</span>
            </div>
            <div style={{ height: '4px', background: 'rgba(255,100,100,0.1)', borderRadius: '2px' }}>
                <div style={{ height: '100%', width: `${percentage}%`, background: color, borderRadius: '2px', transition: 'width 0.3s ease' }} />
            </div>
        </div>
    );
};

const ConfidenceTrendGraph = ({ data = [] }) => {
    const points = data.slice(-20);
    if (points.length < 2) {
        return <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '11px' }}>AWAITING DATA...</div>;
    }

    const svgWidth = 300;
    const svgHeight = 120;
    const pad = { top: 10, bottom: 10, left: 5, right: 5 };
    const gw = svgWidth - pad.left - pad.right;
    const gh = svgHeight - pad.top - pad.bottom;

    const path = points.map((p, i) => {
        const x = pad.left + (i / (points.length - 1)) * gw;
        const y = pad.top + gh * (1 - p.value);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');

    const lastVal = points[points.length - 1].value;

    return (
        <div style={{ flex: 1, position: 'relative' }}>
            <svg width="100%" height="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="xMidYMid meet">
                <path d={path} fill="none" stroke="#FF8B8B" strokeWidth="2" strokeLinejoin="round" />
                <circle cx={pad.left + gw} cy={pad.top + gh * (1 - lastVal)} r="3" fill="#FF8B8B" />
            </svg>
            <span style={{ position: 'absolute', top: '5px', right: '5px', color: '#FF8B8B', fontSize: '12px', fontWeight: 700 }}>
                {(lastVal * 100).toFixed(0)}%
            </span>
        </div>
    );
};

export default Infrared;
