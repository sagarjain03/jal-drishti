import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import { useSystemState } from '../context/SystemStateContext';
import { useStream } from '../context/StreamContext';
import FusionTransparencyPanel from './FusionTransparencyPanel';

const TacticalNav = ({ connectionStatus = 'disconnected', collapsed, onToggle }) => {
    const [showLogs, setShowLogs] = useState(false);
    const [showFusion, setShowFusion] = useState(false);
    
    // Pull live sensor health from centralized state
    let sonarHealth = 0.95;
    let irHealth = 0.92;
    let cameraOnline = connectionStatus === 'connected';

    // Pull events from stream context
    const { events = [] } = useStream();

    // Pull fusion data for the panel
    const { 
        pipelineBreakdown, 
        isVolatile, 
        lowSignalReliability, 
        safeModeReason, 
        similarObjectCount 
    } = useSystemState();

    try {
        const systemState = useSystemState();
        sonarHealth = systemState?.globalState?.sonarData?.sensorHealth ?? 0.95;
        irHealth = systemState?.globalState?.irData?.sensorHealth ?? 0.92;
    } catch {
        // fallback defaults
    }

    const sonarPct = Math.round(sonarHealth * 100);
    const irPct = Math.round(irHealth * 100);
    const navWidth = collapsed ? '54px' : '160px';

    return (
        <nav style={{
            width: navWidth,
            minWidth: navWidth,
            background: '#0D0D0D',
            borderRight: '1px solid #1A1A1A',
            display: 'flex',
            flexDirection: 'column',
            transition: 'width 0.2s ease, min-width 0.2s ease',
            overflow: 'hidden',
            zIndex: 100,
            fontFamily: "'Inter', -apple-system, sans-serif",
            userSelect: 'none',
            position: 'relative' // For absolute positioning of logs if needed, though flex is better
        }}>

            {/* ─── COLLAPSE TOGGLE ──────────────────────── */}
            <div
                onClick={onToggle}
                style={{
                    padding: collapsed ? '12px 0' : '12px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: collapsed ? 'center' : 'space-between',
                    cursor: 'pointer',
                    borderBottom: '1px solid #1A1A1A'
                }}
                title={collapsed ? 'Expand' : 'Collapse'}
            >
                {!collapsed && (
                    <span style={{ color: '#555', fontSize: '10px', fontWeight: 600, letterSpacing: '1.5px' }}>
                        JAL-DRISHTI
                    </span>
                )}
                <span style={{
                    color: '#555',
                    fontSize: '12px',
                    transform: collapsed ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s ease'
                }}>
                    ◂
                </span>
            </div>

            {/* ─── OVERVIEW ─────────────────────────────── */}
            <Section label="OVERVIEW" collapsed={collapsed}>
                <NavItem to="/" icon="◈" label="Integrated" collapsed={collapsed} end
                    statusColor={connectionStatus === 'connected' ? '#4ADE80' : '#EF4444'}
                    statusText={connectionStatus === 'connected' ? 'Active' : 'Offline'} />
            </Section>

            {/* ─── SENSORS ──────────────────────────────── */}
            <Section label="SENSORS" collapsed={collapsed}>
                <NavItem to="/sonar" icon="◎" label="Sonar" collapsed={collapsed}
                    statusColor={healthColor(sonarHealth)} statusText={`${sonarPct}%`} />
                <NavItem to="/infrared" icon="◐" label="Infrared" collapsed={collapsed}
                    statusColor={healthColor(irHealth)} statusText={`${irPct}%`} />
                <NavItem to="/" icon="◉" label="Camera" collapsed={collapsed} disabled
                    statusColor={cameraOnline ? '#4ADE80' : '#555'}
                    statusText={cameraOnline ? 'Live' : 'Offline'} />
            </Section>

            {/* ─── SYSTEM (pinned bottom) ───────────────── */}
            <Section label="SYSTEM" collapsed={collapsed} style={{ marginTop: 'auto' }}>
                {/* Fusion/Config Toggle */}
                <div 
                    onClick={() => setShowFusion(!showFusion)}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: collapsed ? '8px 0' : '7px 14px',
                        justifyContent: collapsed ? 'center' : 'flex-start',
                        cursor: 'pointer',
                        background: showFusion ? '#161616' : 'transparent',
                        borderLeft: showFusion ? '2px solid #888' : '2px solid transparent',
                        transition: 'background 0.15s ease'
                    }}
                    title="Toggle Fusion Config"
                >
                    <Dot color={showFusion ? '#4ADE80' : '#555'} />
                    <span style={{ color: showFusion ? '#ccc' : '#666', fontSize: '13px' }}>⚙</span>
                    {!collapsed && (
                        <span style={{ color: showFusion ? '#ddd' : '#888', fontSize: '12px', flex: 1 }}>Config</span>
                    )}
                </div>

                {/* Logs Toggle */}
                <div 
                    onClick={() => setShowLogs(!showLogs)}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: collapsed ? '8px 0' : '7px 14px',
                        justifyContent: collapsed ? 'center' : 'flex-start',
                        cursor: 'pointer',
                        background: showLogs ? '#161616' : 'transparent',
                        borderLeft: showLogs ? '2px solid #888' : '2px solid transparent',
                        transition: 'background 0.15s ease'
                    }}
                    title="Toggle Logs"
                >
                    <Dot color={showLogs ? '#4ADE80' : '#333'} />
                    <span style={{ color: showLogs ? '#ccc' : '#666', fontSize: '13px' }}>▤</span>
                    {!collapsed && (
                        <span style={{ color: showLogs ? '#ddd' : '#888', fontSize: '12px', flex: 1 }}>Logs</span>
                    )}
                </div>
            </Section>

            {/* ─── FUSION PANEL (Expandable) ──────────────── */}
            {showFusion && !collapsed && (
                <div style={{
                    maxHeight: '40%',
                    background: '#080808',
                    borderTop: '1px solid #1A1A1A',
                    overflowY: 'auto',
                    padding: '0'
                }}>
                    <div style={{ padding: '6px 14px', background: '#0D0D0D', borderBottom: '1px solid #1A1A1A' }}>
                         <span style={{ color: '#444', fontSize: '9px', fontWeight: 700, letterSpacing: '1px' }}>FUSION PIPELINE</span>
                    </div>
                    <div style={{ padding: '8px' }}>
                        <FusionTransparencyPanel
                            breakdown={pipelineBreakdown}
                            isVolatile={isVolatile}
                            lowSignalReliability={lowSignalReliability}
                            safeModeReason={safeModeReason}
                            similarObjectCount={similarObjectCount}
                            embedded={true} // Prop to simplify if needed
                        />
                    </div>
                </div>
            )}

            {/* ─── LOGS PANEL (Expandable) ──────────────── */}
            <LogPanel isOpen={showLogs && !collapsed} events={events} />

        </nav>
    );
};

/* ─── Log Panel ──────────────────────────────────────────── */
const LogPanel = ({ isOpen, events }) => {
    const listRef = useRef(null);

    // Auto-scroll to bottom on new events
    useEffect(() => {
        if (isOpen && listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    }, [events, isOpen]);

    if (!isOpen) return null;

    return (
        <div style={{
            height: '35%', // 30-40% height as requested
            background: '#080808',
            borderTop: '1px solid #1A1A1A',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
        }}>
            <div style={{
                padding: '6px 14px',
                background: '#0D0D0D',
                borderBottom: '1px solid #1A1A1A',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <span style={{ color: '#444', fontSize: '9px', fontWeight: 700, letterSpacing: '1px' }}>EVENT LOG</span>
                <span style={{ color: '#333', fontSize: '9px' }}>{events.length}</span>
            </div>
            
            <div ref={listRef} style={{
                flex: 1,
                overflowY: 'auto',
                padding: '0',
                display: 'flex',
                flexDirection: 'column'
            }}>
                {events.length === 0 ? (
                    <div style={{ padding: '10px', color: '#333', fontSize: '10px', fontStyle: 'italic', textAlign: 'center' }}>
                        No active events
                    </div>
                ) : (
                    events.map((ev, i) => (
                        <LogItem key={i} event={ev} />
                    ))
                )}
            </div>
        </div>
    );
};

const LogItem = ({ event }) => {
    // Subtle color coding
    const color = event.severity === 'danger' ? '#991B1B' : 
                  event.severity === 'warning' ? '#B45309' : 
                  event.severity === 'success' ? '#15803D' : '#333';
                  
    const textColor = event.severity === 'danger' ? '#FCA5A5' : 
                      event.severity === 'warning' ? '#FDBA74' : 
                      event.severity === 'success' ? '#86EFAC' : '#888';

    return (
        <div style={{
            padding: '6px 14px',
            borderBottom: '1px solid #111',
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
            background: 'transparent'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#444', fontSize: '9px', fontFamily: 'monospace' }}>{event.timestamp?.split(' ')[1] || '00:00:00'}</span>
                {event.severity !== 'info' && (
                    <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: color }} />
                )}
            </div>
            <span style={{ color: textColor, fontSize: '10px', lineHeight: '1.3' }}>
                {event.message}
            </span>
        </div>
    );
};

/* ─── Section ────────────────────────────────────────────── */

const Section = ({ label, collapsed, children, style = {} }) => (
    <div style={{ padding: '6px 0', borderBottom: '1px solid #141414', ...style }}>
        {!collapsed && (
            <div style={{
                padding: '4px 14px 5px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
            }}>
                <span style={{ color: '#444', fontSize: '10px', fontWeight: 700, letterSpacing: '1.2px' }}>
                    {label}
                </span>
                <div style={{ flex: 1, height: '1px', background: '#1A1A1A' }} />
            </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
            {children}
        </div>
    </div>
);

/* ─── NavItem ────────────────────────────────────────────── */

const NavItem = ({ to, icon, label, collapsed, statusColor, statusText, disabled = false, end = false }) => {
    if (disabled) {
        return (
            <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: collapsed ? '8px 0' : '7px 14px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                opacity: 0.35, cursor: 'default'
            }} title={label}>
                <Dot color={statusColor} />
                <span style={{ color: '#666', fontSize: '13px' }}>{icon}</span>
                {!collapsed && (
                    <>
                        <span style={{ color: '#666', fontSize: '12px', flex: 1 }}>{label}</span>
                        {statusText && <span style={{ color: '#555', fontSize: '10px' }}>{statusText}</span>}
                    </>
                )}
            </div>
        );
    }

    return (
        <NavLink to={to} end={end} title={label}
            style={({ isActive }) => ({
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: collapsed ? '8px 0' : '7px 14px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                textDecoration: 'none',
                background: isActive ? '#161616' : 'transparent',
                borderLeft: isActive ? '2px solid #888' : '2px solid transparent',
                transition: 'background 0.15s ease'
            })}
        >
            {({ isActive }) => (
                <>
                    <Dot color={isActive ? statusColor : '#555'} />
                    <span style={{ color: isActive ? '#ccc' : '#777', fontSize: '13px' }}>{icon}</span>
                    {!collapsed && (
                        <>
                            <span style={{
                                color: isActive ? '#ddd' : '#888',
                                fontSize: '12px',
                                fontWeight: isActive ? 600 : 400,
                                flex: 1
                            }}>
                                {label}
                            </span>
                            {statusText && (
                                <span style={{
                                    color: isActive ? statusColor : '#555',
                                    fontSize: '10px',
                                    fontWeight: 600
                                }}>
                                    {statusText}
                                </span>
                            )}
                        </>
                    )}
                </>
            )}
        </NavLink>
    );
};

/* ─── Dot ────────────────────────────────────────────────── */

const Dot = ({ color }) => (
    <span style={{
        width: '6px', height: '6px', minWidth: '6px',
        borderRadius: '50%', background: color, display: 'inline-block'
    }} />
);

/* ─── Helper ─────────────────────────────────────────────── */

function healthColor(h) {
    if (h >= 0.8) return '#4ADE80';
    if (h >= 0.5) return '#f97316';
    return '#EF4444';
}

export default TacticalNav;
