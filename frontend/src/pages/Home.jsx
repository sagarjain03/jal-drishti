import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import RawFeedPanel from '../components/RawFeedPanel';
import SafeModeOverlay from '../components/SafeModeOverlay';
import DetectionOverlay from '../components/DetectionOverlay';
import MaximizedPanel from '../components/MaximizedPanel';
import AlertPanel from '../components/AlertPanel';
import MetricsPanel from '../components/MetricsPanel';
import SensorStatusPanel from '../components/SensorStatusPanel';
import InputSourceToggle from '../components/InputSourceToggle';
import LastAlertSnapshot from '../components/LastAlertSnapshot';
import OperatorActionPanel from '../components/OperatorActionPanel';
import SnapshotModal from '../components/SnapshotModal';
import ThreatTimelineStrip from '../components/ThreatTimelineStrip';
import { useStream } from '../context/StreamContext';
import { useSystemState } from '../context/SystemStateContext';

const MAX_HISTORY_POINTS = 60;

const Home = () => {
    // Stream data
    const {
        addEvent,
        events,
        systemStatus,
        fps,
        connectionStatus
    } = useStream();

    // Centralized system state
    const {
        globalState,
        pipelineBreakdown,
        alerts,
        temporalBuffer,
        similarObjectCount,
        lowSignalReliability,
        isVolatile,
        safeModeReason
    } = useSystemState();

    const { displayFrame, inputSource, setInputSource, navCollapsed } = useOutletContext();

    // Local UI state
    const [maximizedPanel, setMaximizedPanel] = useState(null);
    const [snapshotModal, setSnapshotModal] = useState({
        isOpen: false,
        imageData: null,
        timestamp: '',
        alertType: ''
    });
    const [lastAlertSnapshot, setLastAlertSnapshot] = useState(null);

    // ===== FIX 1: Accumulate FPS & Latency history for MetricsPanel graphs =====
    const [fpsHistory, setFpsHistory] = useState([]);
    const [latencyHistory, setLatencyHistory] = useState([]);
    const fpsRef = useRef(fps);
    const latencyRef = useRef(displayFrame?.system?.latency_ms ?? 0);

    // Keep refs in sync without restarting the interval
    useEffect(() => { fpsRef.current = fps; }, [fps]);
    useEffect(() => { latencyRef.current = displayFrame?.system?.latency_ms ?? 0; }, [displayFrame?.system?.latency_ms]);

    // Stable interval — deps are [] so it never restarts
    useEffect(() => {
        const interval = setInterval(() => {
            setFpsHistory(prev => {
                const next = [...prev, fpsRef.current || 0];
                return next.length > MAX_HISTORY_POINTS ? next.slice(-MAX_HISTORY_POINTS) : next;
            });
            setLatencyHistory(prev => {
                const next = [...prev, latencyRef.current || 0];
                return next.length > MAX_HISTORY_POINTS ? next.slice(-MAX_HISTORY_POINTS) : next;
            });
        }, 1000);
        return () => clearInterval(interval);
    }, []); // Stable — never restarts

    // ===== FIX 2: Auto-capture Last Alert Snapshot on alert state =====
    const prevStateRef = useRef(null);
    useEffect(() => {
        const currentState = globalState.state;
        const isAlertState = currentState === 'CONFIRMED_THREAT' || currentState === 'POTENTIAL_ANOMALY';
        const wasAlertState = prevStateRef.current === 'CONFIRMED_THREAT' || prevStateRef.current === 'POTENTIAL_ANOMALY';

        if (isAlertState && !wasAlertState && displayFrame?.image_data) {
            setLastAlertSnapshot({
                imageData: displayFrame.image_data,
                timestamp: new Date().toLocaleTimeString('en-US', {
                    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
                }),
                alertType: currentState
            });
        }
        prevStateRef.current = currentState;
    }, [globalState.state, displayFrame?.image_data]);

    // ===== FIX 3: Throttle BOTH displayFrame AND globalState for side panels =====
    const [throttledFrame, setThrottledFrame] = useState(displayFrame);
    const [throttledGlobalState, setThrottledGlobalState] = useState(globalState);
    const throttleTimerRef = useRef(null);
    const latestFrameRef = useRef(displayFrame);
    const latestGlobalRef = useRef(globalState);

    useEffect(() => {
        latestFrameRef.current = displayFrame;
        latestGlobalRef.current = globalState;
        if (!throttleTimerRef.current) {
            // Immediately update on first call, then throttle subsequent ones
            setThrottledFrame(displayFrame);
            setThrottledGlobalState(globalState);
            throttleTimerRef.current = setTimeout(() => {
                setThrottledFrame(latestFrameRef.current);
                setThrottledGlobalState(latestGlobalRef.current);
                throttleTimerRef.current = null;
            }, 2000); // Update side panels max every 2 seconds
        }
    }, [displayFrame, globalState]);

    // Cleanup throttle timer on unmount
    useEffect(() => {
        return () => {
            if (throttleTimerRef.current) clearTimeout(throttleTimerRef.current);
        };
    }, []);

    // Handlers
    const handleCaptureSnapshot = (e) => {
        e?.stopPropagation();
        if (displayFrame.image_data) {
            setSnapshotModal({
                isOpen: true,
                imageData: displayFrame.image_data,
                timestamp: new Date().toLocaleString(),
                alertType: displayFrame.state
            });
        }
    };

    // Dynamic widths — Center must be widest. Side panels are compact.
    const leftWidth = navCollapsed ? '265px' : '220px';
    const rightWidth = navCollapsed ? '285px' : '240px';

    return (
        <div className="home-dashboard" style={{ 
            height: '100%', 
            display: 'grid', 
            gridTemplateColumns: `${leftWidth} 1fr ${rightWidth}`,
            gap: '10px',
            padding: '10px',
            background: '#050505',
            overflow: 'hidden',
            transition: 'grid-template-columns 0.2s ease'
        }}>
            
            {/* --- COLUMN 1: LEFT PANEL --- */}
            <div className="left-panel" style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '8px', 
                overflowY: 'auto',
                paddingRight: '4px',
                alignContent: 'flex-start'
            }}>
                <div style={{ flex: '0 0 auto' }}>
                    <SensorStatusPanel
                        sensors={throttledFrame.sensors}
                        fusionState={throttledFrame.fusion_state}
                        fusionMessage={throttledFrame.fusion_message}
                        timelineMessages={throttledFrame.timeline_messages}
                    />
                </div>
                <div style={{ flex: '0 0 auto' }}>
                    <InputSourceToggle
                        currentSource={inputSource}
                        sourceState={displayFrame.sourceState || 'IDLE'}
                        onToggle={(source, state) => setInputSource(source)}
                    />
                </div>
            </div>

            {/* --- COLUMN 2: CENTER TACTICAL AREA --- */}
            <div className="center-panel" style={{ 
                display: 'grid',
                gridTemplateRows: '320px auto 1fr', 
                gap: '10px',
                minWidth: 0
            }}>
                
                {/* ROW 1: FEED GRID */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', height: '100%' }}>
                    {/* Raw Feed Panel */}
                    <div className="video-panel clickable" onClick={() => setMaximizedPanel('raw')} style={{ position: 'relative', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid #222' }}>
                        <div className="video-header" style={{ position: 'absolute', top: 0, left: 0, right: 0, padding: '8px', background: 'linear-gradient(to bottom, rgba(0,0,0,0.9), transparent)', zIndex: 10, display: 'flex', justifyContent: 'space-between' }}>
                            <h3 style={{ margin: 0, fontSize: '11px', color: '#ccc', fontWeight: 600, letterSpacing: '0.5px' }}>RAW SENSOR FEED</h3>
                            <button className="expand-btn">⛶</button>
                        </div>
                        <div className="video-content" style={{ width: '100%', height: '100%' }}>
                            <RawFeedPanel />
                            <SafeModeOverlay isActive={systemStatus.inSafeMode} message={systemStatus.message} cause={systemStatus.cause} />
                        </div>
                    </div>

                    {/* Enhanced Feed Panel */}
                    <div className="video-panel clickable" onClick={() => setMaximizedPanel('enhanced')} style={{ position: 'relative', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid #222' }}>
                        <div className="video-header" style={{ position: 'absolute', top: 0, left: 0, right: 0, padding: '8px', background: 'linear-gradient(to bottom, rgba(0,0,0,0.9), transparent)', zIndex: 10, display: 'flex', justifyContent: 'space-between' }}>
                            <h3 style={{ margin: 0, fontSize: '11px', color: '#ccc', fontWeight: 600, letterSpacing: '0.5px' }}>AI ENHANCED FEED</h3>
                            <div style={{ display: 'flex', gap: '6px' }}>
                                <button onClick={handleCaptureSnapshot}>📸</button>
                                <button>⛶</button>
                            </div>
                        </div>
                        <div className="video-content" style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={displayFrame.image_data || "https://placehold.co/640x480/0A0A0A/737373?text=Awaiting+Signal"}
                                alt="Enhanced Feed"
                                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                            />
                            {displayFrame.detections && (
                                <DetectionOverlay
                                    detections={displayFrame.detections}
                                    systemState={displayFrame.state}
                                    width={640}
                                    height={480}
                                />
                            )}
                            <SafeModeOverlay isActive={systemStatus.inSafeMode} message={systemStatus.message} cause={systemStatus.cause} />
                        </div>
                    </div>
                </div>

                {/* ROW 2: THREAT TIMELINE + RISK INDICATOR */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <ThreatTimelineStrip />
                    
                    <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        background: '#111',
                        border: '1px solid #222',
                        borderRadius: '4px',
                        padding: '5px 12px'
                    }}>
                        <span style={{ fontSize: '11px', color: '#888', fontWeight: 600, letterSpacing: '1px' }}>SYSTEM ALERT LOG</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '10px', color: '#666' }}>CURRENT RISK LEVEL:</span>
                            <span style={{ 
                                fontSize: '11px', 
                                fontWeight: 800, 
                                color: globalState.state === 'SAFE_MODE' ? '#22C55E' : 
                                       globalState.state === 'POTENTIAL_THREAT' ? '#F97316' : 
                                       globalState.state === 'CONFIRMED_THREAT' ? '#EF4444' : '#EAB308',
                                letterSpacing: '0.5px'
                            }}>
                                {globalState.state?.replace('_', ' ') || 'UNKNOWN'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* ROW 3: ALERTS — fills remaining space */}
                <div className="alert-panel-wrapper" style={{ 
                    flex: 1, 
                    background: '#0D0D0D', 
                    borderRadius: '6px', 
                    border: '1px solid #1A1A1A',
                    overflow: 'hidden',
                    minHeight: 0
                }}>
                    <AlertPanel
                        alerts={alerts}
                        lowSignalReliability={lowSignalReliability}
                        safeModeReason={safeModeReason}
                        currentState={globalState.state}
                        addEvent={addEvent}
                    />
                </div>
            </div>

            {/* --- COLUMN 3: RIGHT PANEL --- */}
            <div className="right-panel" style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '8px', 
                overflowY: 'auto',
                paddingRight: '4px' 
            }}>
                <div style={{ flex: '0 0 auto' }}>
                    <OperatorActionPanel
                        threatPriority={throttledGlobalState.threatPriority || throttledFrame.threat_priority}
                        signature={throttledGlobalState.signature || throttledFrame.signature}
                        riskScore={throttledGlobalState.riskScore || throttledFrame.risk_score}
                        fusionState={throttledGlobalState.fusionState || throttledFrame.fusion_state}
                        seenBefore={throttledGlobalState.seenBefore ?? throttledFrame.seen_before}
                        occurrenceCount={throttledGlobalState.occurrenceCount || throttledFrame.occurrence_count}
                        explainability={throttledGlobalState.explainability || throttledFrame.explainability}
                        onDecision={(decision) => console.log('Operator Decision:', decision)}
                    />
                </div>
                
                <div style={{ flex: '0 0 auto', background: '#0D0D0D', borderRadius: '6px', border: '1px solid #1A1A1A', padding: '6px' }}>
                    <MetricsPanel
                        fpsHistory={fpsHistory}
                        latencyHistory={latencyHistory}
                        inSafeMode={systemStatus.inSafeMode}
                        safeModeStartTime={null}
                        currentFps={fps}
                        latency={displayFrame.system?.latency_ms}
                        connectionStatus={connectionStatus}
                        systemState={globalState.state}
                    />
                </div>

                <div style={{ flex: '0 0 auto' }}>
                    <LastAlertSnapshot snapshot={lastAlertSnapshot} />
                </div>
            </div>

            {/* Modals */}
            <MaximizedPanel
                isOpen={maximizedPanel === 'raw'}
                onClose={() => setMaximizedPanel(null)}
                title="Raw Feed (Sensor)"
                badge="RAW"
            >
                <RawFeedPanel />
                <SafeModeOverlay isActive={systemStatus.inSafeMode} message={systemStatus.message} cause={systemStatus.cause} />
            </MaximizedPanel>

            <MaximizedPanel
                isOpen={maximizedPanel === 'enhanced'}
                onClose={() => setMaximizedPanel(null)}
                title="Enhanced Feed"
                badge="AI ENHANCED"
            >
                <img
                    src={displayFrame.image_data || "https://placehold.co/640x480/0A0A0A/737373?text=Awaiting+Signal"}
                    alt="Enhanced Feed"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
                <SafeModeOverlay isActive={systemStatus.inSafeMode} message={systemStatus.message} cause={systemStatus.cause} />
            </MaximizedPanel>

            <SnapshotModal
                isOpen={snapshotModal.isOpen}
                onClose={() => setSnapshotModal({ isOpen: false, imageData: null, type: null })}
                imageData={snapshotModal.imageData}
                timestamp={snapshotModal.timestamp}
                alertType={snapshotModal.alertType}
            />
        </div>
    );
};

export default Home;
