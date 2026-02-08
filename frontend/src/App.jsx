import React, { useRef, useEffect, useState } from 'react';
import RawFeedPanel from './components/RawFeedPanel';

import StatusBar from './components/StatusBar';
import AlertPanel from './components/AlertPanel';
import ConnectionOverlay from './components/ConnectionOverlay';
import SafeModeOverlay from './components/SafeModeOverlay';
import EventTimeline from './components/EventTimeline';
import DetectionOverlay from './components/DetectionOverlay';
import MaximizedPanel from './components/MaximizedPanel';
import MetricsPanel from './components/MetricsPanel';
import SnapshotModal from './components/SnapshotModal';
import LastAlertSnapshot from './components/LastAlertSnapshot';
import InputSourceToggle from './components/InputSourceToggle';
import ConnectedViewers from './components/ConnectedViewers';
import RiskScoreCircle from './components/RiskScoreCircle';
import SensorStatusPanel from './components/SensorStatusPanel';
import OperatorActionPanel from './components/OperatorActionPanel';

import useLiveStream from './hooks/useLiveStream';
import { SYSTEM_STATES, CONNECTION_STATES, INPUT_SOURCES, EVENT_TYPES, SOURCE_STATES } from './constants';
import './App.css';

/**
 * TEST MODE TOGGLE
 */
const USE_FAKE_STREAM = false;

/**
 * Format milliseconds to HH:MM:SS
 */
const formatUptime = (ms) => {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
};

/**
 * App Component (Phase 4 Redesign)
 * 3-Column Layout: Left (Controls), Center (Monitoring), Right (History/Admin)
 */
function App() {
  const [inputSource, setInputSource] = useState('video');
  const [sourceState, setSourceState] = useState(SOURCE_STATES.IDLE);
  const [maximizedPanel, setMaximizedPanel] = useState(null);
  const [showRecoveryFlash, setShowRecoveryFlash] = useState(false);
  const prevSafeModeRef = useRef(false);

  // Metrics history
  const [fpsHistory, setFpsHistory] = useState([]);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [safeModeStartTime, setSafeModeStartTime] = useState(null);

  // Smart Uptime
  const [uptime, setUptime] = useState('00:00:00');

  useEffect(() => {
    const currentBuildTime = __BUILD_TIMESTAMP__;
    const storedBuildTime = localStorage.getItem('jalDrishtiBuildTimestamp');
    let totalActiveTime = 0;

    if (!storedBuildTime || parseInt(storedBuildTime) !== currentBuildTime) {
      localStorage.setItem('jalDrishtiBuildTimestamp', currentBuildTime.toString());
      localStorage.setItem('jalDrishtiTotalActiveTime', '0');
    } else {
      const savedTime = localStorage.getItem('jalDrishtiTotalActiveTime');
      totalActiveTime = savedTime ? parseInt(savedTime, 10) : 0;
    }

    const interval = setInterval(() => {
      totalActiveTime += 1000;
      setUptime(formatUptime(totalActiveTime));
      localStorage.setItem('jalDrishtiTotalActiveTime', totalActiveTime.toString());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Snapshot modal
  const [snapshotModal, setSnapshotModal] = useState({
    isOpen: false,
    imageData: null,
    timestamp: '',
    alertType: ''
  });

  const [lastAlertSnapshot, setLastAlertSnapshot] = useState(null);

  // Live Stream Hook
  const liveStreamData = useLiveStream(!USE_FAKE_STREAM);
  const {
    frame = null,
    fps = 0,
    connectionStatus = CONNECTION_STATES.CONNECTED,
    reconnectAttempt = 0,
    lastValidFrame = null,
    manualReconnect = () => { },
    systemStatus = { inSafeMode: false, message: null, cause: null },
    events = [],
    addEvent = () => { },
    wsRef = { current: null },
    // Mock risk values for circle
    riskLevel = 0 // Assuming hook might provide this later, or we derive it
  } = liveStreamData;

  const prevStateRef = useRef(SYSTEM_STATES.SAFE_MODE);

  // Determine display frame
  const displayFrame = frame || lastValidFrame || {
    state: SYSTEM_STATES.SAFE_MODE,
    max_confidence: 0,
    detections: [],
    image_data: null,
    system: { fps: null, latency_ms: null },
    risk_score: 0
  };

  // Track state changes
  useEffect(() => {
    const currentState = displayFrame.state;
    if (prevStateRef.current !== currentState && addEvent) {
      const stateLabels = {
        [SYSTEM_STATES.CONFIRMED_THREAT]: 'THREAT CONFIRMED',
        [SYSTEM_STATES.POTENTIAL_ANOMALY]: 'Potential Anomaly Detected',
        [SYSTEM_STATES.SAFE_MODE]: 'System Normal'
      };
      const severity = currentState === SYSTEM_STATES.CONFIRMED_THREAT ? 'danger' :
        currentState === SYSTEM_STATES.POTENTIAL_ANOMALY ? 'warning' : 'success';

      addEvent(EVENT_TYPES.STATE_CHANGE, stateLabels[currentState] || 'State Changed', severity);

      if (currentState === SYSTEM_STATES.CONFIRMED_THREAT || currentState === SYSTEM_STATES.POTENTIAL_ANOMALY) {
        if (displayFrame.image_data) {
          setLastAlertSnapshot({
            imageData: displayFrame.image_data,
            timestamp: new Date().toLocaleTimeString(),
            alertType: currentState
          });
        }
      }
      prevStateRef.current = currentState;
    }
  }, [displayFrame.state, displayFrame.image_data, addEvent]);

  // Flash effect
  useEffect(() => {
    const wasInSafeMode = prevSafeModeRef.current;
    const isInSafeMode = systemStatus.inSafeMode;
    if (wasInSafeMode && !isInSafeMode) {
      setShowRecoveryFlash(true);
      setSafeModeStartTime(null);
      setTimeout(() => setShowRecoveryFlash(false), 1000);
    }
    if (!wasInSafeMode && isInSafeMode) {
      setSafeModeStartTime(Date.now());
    }
    prevSafeModeRef.current = isInSafeMode;
  }, [systemStatus.inSafeMode]);

  // Metrics history
  useEffect(() => {
    if (fps !== undefined) setFpsHistory(prev => [...prev.slice(-59), fps]);
    const latency = displayFrame.system?.latency_ms;
    if (latency !== null && latency !== undefined) setLatencyHistory(prev => [...prev.slice(-59), latency]);
  }, [fps, displayFrame.system?.latency_ms]);

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

  // RENDER: 3-COLUMN LAYOUT (Phase-4)
  return (
    <div className={`app-container ${systemStatus.inSafeMode ? 'safe-mode-active' : ''} ${showRecoveryFlash ? 'recovery-flash' : ''}`}>

      {/* Status Bar */}
      <StatusBar
        systemState={displayFrame.state}
        maxConfidence={displayFrame.max_confidence}
        latencyMs={displayFrame.system?.latency_ms}
        renderFps={fps}
        mlFps={displayFrame.system?.fps}
        connectionStatus={connectionStatus}
        inputSource={inputSource}
        uptime={uptime}
      />

      {/* Main Content Areas */}
      <div className="main-content">

        {/* --- COLUMN 1: LEFT SIDEBAR (Controls) --- */}
        <div className="left-sidebar">
          {/* Phase-3: Connected Viewers Panel (Moved from Right/Bottom) */}
          {/* Wait, user said "remove scroll completely create a right sidepanel too and add remaining components there"
               Left: Input Source, Last Alert Snapshot
               Right: Viewers, Timeline
           */}

          {/* Sensor Status (Restored) */}
          <SensorStatusPanel
            sensors={displayFrame.sensors}
            fusionState={displayFrame.fusion_state}
            fusionMessage={displayFrame.fusion_message}
            timelineMessages={displayFrame.timeline_messages}
          />

          {/* Input Source Toggle */}
          <InputSourceToggle
            currentSource={inputSource}
            sourceState={sourceState}
            onToggle={(source, state) => {
              setInputSource(source);
              if (state) setSourceState(state);
            }}
            wsConnection={wsRef.current}
            onReset={() => console.log('[App] Source switched')}
          />

          {/* Last Alert Snapshot */}
          <LastAlertSnapshot snapshot={lastAlertSnapshot} />
        </div>

        {/* --- COLUMN 2: CENTER DASHBOARD (2x2 Grid) --- */}
        <div className="center-dashboard">

          {/* TOP ROW: VIDEOS */}
          <div className="video-grid-row">
            {/* Raw Feed Panel */}
            <div
              className="video-panel clickable"
              onClick={() => setMaximizedPanel('raw')}
            >
              <div className="video-header">
                <h3 className="video-title">Raw Feed (Sensor)</h3>
                <div className="video-header-controls">
                  <span className="badge-live" style={{ background: '#333' }}>RAW</span>
                  <button className="expand-btn" title="Expand">⛶</button>
                </div>
              </div>
              <div className="video-content">
                <RawFeedPanel />
                <SafeModeOverlay
                  isActive={systemStatus.inSafeMode}
                  message={systemStatus.message}
                  cause={systemStatus.cause}
                />
              </div>
            </div>

            {/* Enhanced Feed Panel */}
            <div
              className="video-panel clickable"
              onClick={() => setMaximizedPanel('enhanced')}
            >
              <div className="video-header">
                <h3 className="video-title">Enhanced Feed</h3>
                <div className="video-header-controls">
                  <span className="badge-live">AI ENHANCED</span>
                  <button className="capture-btn" onClick={handleCaptureSnapshot} title="Capture">📸</button>
                  <button className="expand-btn" title="Expand">⛶</button>
                </div>
              </div>
              <div className="video-content">
                <img
                  src={displayFrame.image_data || "https://placehold.co/640x480/0A0A0A/737373?text=Awaiting+Signal"}
                  alt="Enhanced Feed"
                  className="video-feed"
                />
                {displayFrame.detections && (
                  <DetectionOverlay
                    detections={displayFrame.detections}
                    systemState={displayFrame.state}
                    width={640}
                    height={480}
                  />
                )}
                <SafeModeOverlay
                  isActive={systemStatus.inSafeMode}
                  message={systemStatus.message}
                  cause={systemStatus.cause}
                />
              </div>
            </div>
          </div>

          {/* BOTTOM ROW: LOGS & METRICS */}
          <div className="data-grid-row">
            <div className="alert-panel-wrapper">
              <AlertPanel
                currentState={displayFrame.state}
                detections={displayFrame.detections}
                maxConfidence={displayFrame.max_confidence}
                addEvent={addEvent}
              />
            </div>
            <div className="metrics-panel-wrapper">
              <MetricsPanel
                fpsHistory={fpsHistory}
                latencyHistory={latencyHistory}
                inSafeMode={systemStatus.inSafeMode}
                safeModeStartTime={safeModeStartTime}
                currentFps={fps}
                latency={displayFrame.system?.latency_ms}
                connectionStatus={connectionStatus}
                systemState={displayFrame.state}
              />
            </div>
          </div>

          {/* CENTER OVERLAY: RISK SCORE */}
          <div className="center-overlay">
            <RiskScoreCircle
              systemState={displayFrame.state}
              confidence={displayFrame.max_confidence || 0}
              isSafeMode={systemStatus.inSafeMode}
            />
          </div>

        </div>

        {/* --- COLUMN 3: RIGHT SIDEBAR --- */}
        <div className="right-sidebar">
          <ConnectedViewers isOperator={true} />

          {/* Operator Actions (Restored) */}
          <OperatorActionPanel
            threatPriority={displayFrame.threat_priority}
            signature={displayFrame.signature}
            riskScore={displayFrame.risk_score}
            fusionState={displayFrame.fusion_state}
            seenBefore={displayFrame.seen_before}
            occurrenceCount={displayFrame.occurrence_count}
            explainability={displayFrame.explainability}
            onDecision={(decision) => console.log('[M4] Operator Decision:', decision)}
          />

          <div className="timeline-container" style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <EventTimeline events={events} />
          </div>
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
        <SafeModeOverlay
          isActive={systemStatus.inSafeMode}
          message={systemStatus.message}
          cause={systemStatus.cause}
        />
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
          className="video-feed"
        />
        {displayFrame.detections && (
          <DetectionOverlay
            detections={displayFrame.detections}
            systemState={displayFrame.state}
            width={640}
            height={480}
          />
        )}
        <SafeModeOverlay
          isActive={systemStatus.inSafeMode}
          message={systemStatus.message}
          cause={systemStatus.cause}
        />
      </MaximizedPanel>

      <SnapshotModal
        isOpen={snapshotModal.isOpen}
        onClose={() => setSnapshotModal({ isOpen: false, imageData: null, type: null })}
        imageData={snapshotModal.imageData}
        timestamp={snapshotModal.timestamp}
        alertType={snapshotModal.alertType}
      />

      <ConnectionOverlay
        connectionStatus={connectionStatus}
        reconnectAttempt={reconnectAttempt}
        onRetry={manualReconnect}
      />
    </div>
  );
}

export default App;
