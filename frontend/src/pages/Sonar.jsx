import React, { useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import SonarRadar from '../components/SonarRadar';
import SonarTemporalGraph from '../components/SonarTemporalGraph';
import { useSonarEngine } from '../sensors/sonar/sonarEngine';
import { useSystemState } from '../context/SystemStateContext';
import { ZONES, MAX_RANGE } from '../sensors/sonar/sonarConfig';

/**
 * Sonar Page (v3 — Engine-Driven)
 * 
 * All data from useSonarEngine() — zero hardcoded values.
 * Panels: Confidence Composition, Environmental, Doppler/Velocity, 
 *         SNR, Bearing Drift, Persistence Timer
 */
const Sonar = () => {
    const { systemStatus } = useOutletContext();
    const { updateSonarData } = useSystemState();

    // Sonar engine: feeds SystemStateManager via callback
    const sonar = useSonarEngine(useCallback((data) => {
        updateSonarData(data);
    }, [updateSonarData]));

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '15px', gap: '12px', overflowY: 'auto', background: '#09090B' }}>
            {/* PAGE HEADER */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ margin: 0, color: '#A1A1AA', fontSize: '16px', letterSpacing: '2px', fontFamily: '"JetBrains Mono", monospace' }}>
                    SONAR ANALYSIS
                </h2>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <StatusBadge label="SNR" value={`${sonar.snr} dB`} quality={sonar.snrQuality} />
                    <StatusBadge label="HEALTH" value={`${Math.round(sonar.sensorHealth * 100)}%`} quality={sonar.sensorHealth > 0.7 ? 'GOOD' : sonar.sensorHealth > 0.4 ? 'MODERATE' : 'POOR'} />
                    <StatusBadge label="STABILITY" value={sonar.objectStability} quality={sonar.objectStability === 'CONFIRMED' ? 'GOOD' : 'MODERATE'} />
                </div>
            </div>

            {/* TOP ROW: Radar + Temporal Graph */}
            <div style={{ display: 'flex', gap: '12px', flex: 1, minHeight: '350px' }}>
                {/* Sonar Radar */}
                <div style={{ flex: 1, background: '#18181B', borderRadius: '10px', border: '1px solid #27272A', padding: '12px', display: 'flex', flexDirection: 'column' }}>
                    <SectionHeader title="SONAR RADAR" subtitle={`MAX RANGE: ${MAX_RANGE}m`} />
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 0 }}>
                        <SonarRadar detections={sonar.detections} />
                    </div>
                </div>

                {/* Temporal Graph (distance + signal + noise) */}
                <div style={{ flex: 1, background: '#18181B', borderRadius: '10px', border: '1px solid #27272A', padding: '12px', display: 'flex', flexDirection: 'column' }}>
                    <SectionHeader title="TEMPORAL ANALYSIS" subtitle="30s ROLLING BUFFER" />
                    <div style={{ flex: 1 }}>
                        <SonarTemporalGraph timeSeries={sonar.timeSeries} />
                    </div>
                </div>
            </div>

            {/* BOTTOM ROW: Data Panels */}
            <div style={{ display: 'flex', gap: '12px', height: '200px' }}>
                {/* Confidence Composition */}
                <DataPanel title="CONFIDENCE COMPOSITION" style={{ flex: 1 }}>
                    <CompositionBar label="Range" value={sonar.confidenceBreakdown.range} max={0.35} color="#22C55E" />
                    <CompositionBar label="Signal" value={sonar.confidenceBreakdown.signal} max={0.30} color="#3B82F6" />
                    <CompositionBar label="Noise" value={sonar.confidenceBreakdown.noise} max={0.20} color="#EF4444" isNegative />
                    <CompositionBar label="Doppler" value={sonar.confidenceBreakdown.doppler} max={0.15} color="#A855F7" />
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', marginTop: '6px', paddingTop: '6px', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: '#A1A1AA', fontSize: '10px' }}>TOTAL</span>
                        <span style={{ color: '#22C55E', fontSize: '13px', fontWeight: 700 }}>{(sonar.confidence * 100).toFixed(0)}%</span>
                    </div>
                </DataPanel>

                {/* SNR & Doppler */}
                <DataPanel title="SIGNAL / DOPPLER" style={{ flex: 1 }}>
                    <MetricRow label="Signal" value={sonar.signalStrength.toFixed(2)} />
                    <MetricRow label="Noise" value={sonar.noiseLevel.toFixed(2)} warn={sonar.noiseLevel > 0.5} />
                    <MetricRow label="SNR" value={`${sonar.snr} dB`} />
                    <MetricRow label="SNR Quality" value={sonar.snrQuality} />
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', marginTop: '4px', paddingTop: '4px' }}>
                        <MetricRow label="Velocity" value={`${sonar.velocity} m/s`} />
                        <MetricRow label="Direction" value={sonar.velocityDirection} />
                        {sonar.velocityDirection === 'APPROACHING' && (
                            <span style={{ color: '#EF4444', fontSize: '10px', fontWeight: 700 }}>↓ APPROACHING</span>
                        )}
                        {sonar.velocityDirection === 'RECEDING' && (
                            <span style={{ color: '#22C55E', fontSize: '10px', fontWeight: 700 }}>↑ RECEDING</span>
                        )}
                    </div>
                </DataPanel>

                {/* Environmental */}
                <DataPanel title="ENVIRONMENT" style={{ flex: 1 }}>
                    <MetricRow label="Turbidity" value={(sonar.environment?.turbidity * 100).toFixed(0) + '%'} warn={sonar.environment?.turbidity > 0.6} />
                    <MetricRow label="Salinity" value={(sonar.environment?.salinity * 100).toFixed(0) + '%'} />
                    <MetricRow label="Bkg Noise" value={(sonar.environment?.backgroundNoise * 100).toFixed(0) + '%'} warn={sonar.environment?.backgroundNoise > 0.5} />
                    <MetricRow label="Sensor HP" value={`${Math.round((sonar.environment?.sensorHealth || 0) * 100)}%`} warn={sonar.environment?.sensorHealth < 0.5} />
                </DataPanel>

                {/* Bearing Drift & Persistence */}
                <DataPanel title="TRACKING" style={{ flex: 1 }}>
                    <span style={{ color: '#71717A', fontSize: '9px', marginBottom: '4px', display: 'block' }}>BEARING DRIFT</span>
                    {Object.entries(sonar.bearingDrift || {}).map(([label, data]) => (
                        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '2px' }}>
                            <span style={{ color: '#A1A1AA' }}>{label}</span>
                            <span style={{ color: data.trend === 'STABLE' ? '#22C55E' : '#F97316' }}>
                                {data.drift > 0 ? '+' : ''}{data.drift}° {data.trend}
                            </span>
                        </div>
                    ))}
                    <span style={{ color: '#71717A', fontSize: '9px', marginTop: '8px', marginBottom: '4px', display: 'block' }}>PERSISTENCE</span>
                    {Object.entries(sonar.persistence || {}).map(([label, data]) => (
                        <div key={label} style={{ marginBottom: '3px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
                                <span style={{ color: '#A1A1AA' }}>{label}</span>
                                <span style={{ color: data.confirmed ? '#22C55E' : '#F97316', fontWeight: 600 }}>
                                    {data.confirmed ? '✓ CONFIRMED' : `${Math.round(data.progress * 100)}%`}
                                </span>
                            </div>
                            <div style={{ height: '3px', background: '#27272A', borderRadius: '2px', marginTop: '2px' }}>
                                <div style={{ height: '100%', width: `${data.progress * 100}%`, background: data.confirmed ? '#22C55E' : '#F97316', borderRadius: '2px', transition: 'width 0.3s ease' }} />
                            </div>
                        </div>
                    ))}
                </DataPanel>
            </div>
        </div>
    );
};

/* ─── Reusable Sub-Components ──────────────────────────── */

const SectionHeader = ({ title, subtitle }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ color: '#A1A1AA', fontWeight: 700, fontSize: '11px', letterSpacing: '1px' }}>{title}</span>
        <span style={{ color: '#52525B', fontSize: '9px' }}>{subtitle}</span>
    </div>
);

const StatusBadge = ({ label, value, quality }) => {
    const color = quality === 'GOOD' ? '#22C55E' : quality === 'MODERATE' ? '#F97316' : '#EF4444';
    return (
        <div style={{ background: '#18181B', border: `1px solid ${color}30`, borderRadius: '6px', padding: '4px 10px', display: 'flex', gap: '6px', alignItems: 'center' }}>
            <span style={{ color: '#71717A', fontSize: '9px' }}>{label}</span>
            <span style={{ color, fontSize: '11px', fontWeight: 700 }}>{value}</span>
        </div>
    );
};

const DataPanel = ({ title, children, style = {} }) => (
    <div style={{ background: '#18181B', borderRadius: '10px', border: '1px solid #27272A', padding: '10px', display: 'flex', flexDirection: 'column', ...style }}>
        <span style={{ color: '#A1A1AA', fontWeight: 700, fontSize: '10px', letterSpacing: '0.5px', marginBottom: '8px' }}>{title}</span>
        {children}
    </div>
);

const MetricRow = ({ label, value, warn = false }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
        <span style={{ color: '#94A3B8' }}>{label}</span>
        <span style={{ color: warn ? '#F97316' : '#CBD5E1', fontWeight: 500 }}>{value}</span>
    </div>
);

const CompositionBar = ({ label, value, max, color, isNegative = false }) => {
    const percentage = Math.min(100, (value / max) * 100);
    return (
        <div style={{ marginBottom: '5px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '2px' }}>
                <span style={{ color: '#94A3B8' }}>{label} {isNegative && '(−)'}</span>
                <span style={{ color, fontWeight: 600 }}>{value.toFixed(2)}</span>
            </div>
            <div style={{ height: '4px', background: 'rgba(100,140,255,0.1)', borderRadius: '2px' }}>
                <div style={{ height: '100%', width: `${percentage}%`, background: color, borderRadius: '2px', transition: 'width 0.3s ease' }} />
            </div>
        </div>
    );
};

export default Sonar;
