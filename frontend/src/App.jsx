import RawFeedPanel from './components/RawFeedPanel';
import React, { useRef, useEffect, useState } from 'react';
import StatusBar from './components/StatusBar';
import VideoPanel from './components/VideoPanel';
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
import useLiveStream from './hooks/useLiveStream';
import useFakeStream from './hooks/useFakeStream';
import { SYSTEM_STATES, CONNECTION_STATES, INPUT_SOURCES, EVENT_TYPES } from './constants';
import './App.css';

/**
 * TEST MODE TOGGLE
 * 
 * Set to true to use fake stream (frontend-only testing)
 * Set to false to use live stream (requires backend)
 * 
 * For TEST 1-4: Set USE_FAKE_STREAM = true
 * For TEST 5:   Set USE_FAKE_STREAM = false
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
 * App Component (Enhanced - Phase 3)
 * 
 * Main dashboard for Jal-Drishti.
 * Features:
 * - Dark defence-theme UI
 * - Safe Mode visual overlay
 * - Event timeline panel
 * - Enhanced status bar with color coding
 * - Detection visual polish
 * - Click-to-maximize video panels with expand icons
 * - Snapshot capture functionality
 * - System uptime counter (Smart Persistence)
 * - Last alert snapshot panel
 * - Analytical Alert Panel
 * 
 * IMPORTANT: Backend is unchanged. All enhancements are frontend-only.
 */
function App() {
  const [inputSource, setInputSource] = useState(INPUT_SOURCES.DUMMY_VIDEO);

  // Maximize panel state: null, 'raw', or 'enhanced'
  const [maximizedPanel, setMaximizedPanel] = useState(null);

  // Recovery flash animation state
  const [showRecoveryFlash, setShowRecoveryFlash] = useState(false);

  // Previous safe mode state for detecting transitions
  const prevSafeModeRef = useRef(false);

  // Metrics history for graphs (last 60 data points)
  const [fpsHistory, setFpsHistory] = useState([]);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [safeModeStartTime, setSafeModeStartTime] = useState(null);

  // Phase-3: Smart System Uptime
  // Reset on server restart (detected via build timestamp), pause on tab close.
  const [uptime, setUptime] = useState('00:00:00');

  useEffect(() => {
    // 1. Check for server restart
    const currentBuildTime = __BUILD_TIMESTAMP__;
    const storedBuildTime = localStorage.getItem('jalDrishtiBuildTimestamp');

    let totalActiveTime = 0;

    if (!storedBuildTime || parseInt(storedBuildTime) !== currentBuildTime) {
      // Server restarted or first run -> RESET
      localStorage.setItem('jalDrishtiBuildTimestamp', currentBuildTime.toString());
      localStorage.setItem('jalDrishtiTotalActiveTime', '0');
    } else {
      // Same session -> RESUME
      const savedTime = localStorage.getItem('jalDrishtiTotalActiveTime');
      totalActiveTime = savedTime ? parseInt(savedTime, 10) : 0;
    }

    // 2. Start Interval
    const interval = setInterval(() => {
      totalActiveTime += 1000; // increment by 1 second
      setUptime(formatUptime(totalActiveTime));
      localStorage.setItem('jalDrishtiTotalActiveTime', totalActiveTime.toString());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Phase-3: Snapshot modal state
  const [snapshotModal, setSnapshotModal] = useState({
    isOpen: false,
    imageData: null,
    timestamp: '',
    alertType: ''
  });

  // Phase-3: Last alert snapshot
  const [lastAlertSnapshot, setLastAlertSnapshot] = useState(null);

  // Only use live stream when not in fake/test mode
  const liveStreamData = useLiveStream(!USE_FAKE_STREAM);

  // Destructure with defaults for when hook returns null/disconnected state
  const {
    frame = null,

    fps = 0,
    connectionStatus = CONNECTION_STATES.CONNECTED, // When fake, pretend connected
    reconnectAttempt = 0,
    lastValidFrame = null,
    manualReconnect = () => { },
    // New enhanced exports
    systemStatus = { inSafeMode: false, message: null, cause: null },
    events = [],
    addEvent = () => { }
  } = liveStreamData;

  // Use ref for previous state tracking (not useState - avoids re-renders)
  const prevStateRef = useRef(SYSTEM_STATES.SAFE_MODE);

  // Determine which frame to display (current or last valid on disconnect)
  const displayFrame = frame || lastValidFrame || {
    state: SYSTEM_STATES.SAFE_MODE,
    max_confidence: 0,
    detections: [],
    visibility_score: 0,
    image_data: null,
    system: { fps: null, latency_ms: null }
  };

  // Track state changes and add to event timeline
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

      // Phase-3: Auto-capture snapshot on alert
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

  // Add detection events when new detections appear
  useEffect(() => {
    if (displayFrame.detections && displayFrame.detections.length > 0 && addEvent) {
      const detectionCount = displayFrame.detections.length;
      // Only log significant detections (high confidence)
      const highConfidence = displayFrame.detections.filter(d => d.confidence > 0.5);
      if (highConfidence.length > 0) {
        // Throttle: don't add detection events too frequently
        // (handled by the hook's deduplication if needed)
      }
    }
  }, [displayFrame.detections, addEvent]);

  // Track safe mode transitions for recovery flash effect
  useEffect(() => {
    const wasInSafeMode = prevSafeModeRef.current;
    const isInSafeMode = systemStatus.inSafeMode;

    // Detect recovery: was in safe mode, now not
    if (wasInSafeMode && !isInSafeMode) {
      setShowRecoveryFlash(true);
      setSafeModeStartTime(null);
      const timer = setTimeout(() => setShowRecoveryFlash(false), 1000);
      return () => clearTimeout(timer);
    }

    // Detect entering safe mode
    if (!wasInSafeMode && isInSafeMode) {
      setSafeModeStartTime(Date.now());
    }

    prevSafeModeRef.current = isInSafeMode;
  }, [systemStatus.inSafeMode]);

  // Update metrics history when new frame data arrives
  useEffect(() => {
    // Update FPS history
    if (fps !== undefined) {
      setFpsHistory(prev => [...prev.slice(-59), fps]);
    }

    // Update latency history
    const latency = displayFrame.system?.latency_ms;
    if (latency !== null && latency !== undefined) {
      setLatencyHistory(prev => [...prev.slice(-59), latency]);
    }
  }, [fps, displayFrame.system?.latency_ms]);

  // Phase-3: Handle snapshot capture
  const handleCaptureSnapshot = (e) => {
    e.stopPropagation(); // Prevent panel click
    if (displayFrame.image_data) {
      setSnapshotModal({
        isOpen: true,
        imageData: displayFrame.image_data,
        timestamp: new Date().toLocaleString(),
        alertType: displayFrame.state
      });
    }
  };

  const closeSnapshotModal = () => {
    setSnapshotModal({
      isOpen: false,
      imageData: null,
      timestamp: '',
      alertType: ''
    });
  };

  // Handle Input Source Toggle
  const handleSourceToggle = (source) => {
    // In a real app, this would switch backend streams
    // For now, we update frontend state to show the toggle works
    setInputSource(source === 'camera' ? 'camera' : INPUT_SOURCES.DUMMY_VIDEO);
  };

  return (
    <div className={`app-container ${systemStatus.inSafeMode ? 'safe-mode-active' : ''} ${showRecoveryFlash ? 'recovery-flash' : ''}`}>
      {/* Test Mode Indicator */}
      {USE_FAKE_STREAM && (
        <div className="test-mode-banner">
          🧪 TEST MODE: Using Fake Stream (state cycles every ~10s)
        </div>
      )}

      {/* Enhanced Status Bar with Input Source and Uptime */}
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

      <main className="main-content">
        {/* Left Sidebar */}
        <div className="left-sidebar">
          <EventTimeline events={events} />
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
          {/* Phase-3: Last Alert Snapshot Panel */}
          <LastAlertSnapshot snapshot={lastAlertSnapshot} />

          {/* Phase-3: Input Source Toggle (Fills empty space) */}
          <InputSourceToggle
            currentSource={inputSource === 'camera' ? 'camera' : 'video'}
            onToggle={handleSourceToggle}
          />
        </div>

        {/* Real-time RAW Feed - Click to maximize */}
        <div
          className="video-panel clickable"
          onClick={() => setMaximizedPanel('raw')}
        >
          <div className="video-header">
            <h3 className="video-title">Raw Feed (Sensor)</h3>
            <div className="video-header-controls">
              <span className="badge-live" style={{ background: '#333' }}>RAW</span>
              <button
                className="expand-btn"
                onClick={(e) => { e.stopPropagation(); setMaximizedPanel('raw'); }}
                title="Expand"
              >⛶</button>
            </div>
          </div>
          <div className="video-content">
            <RawFeedPanel />
            <div className="maximize-hint">Click to expand</div>

            {/* Safe Mode Overlay on Raw Feed */}
            <SafeModeOverlay
              isActive={systemStatus.inSafeMode}
              message={systemStatus.message}
              cause={systemStatus.cause}
            />
          </div>
        </div>

        {/* Enhanced Video Panel - Click to maximize */}
        <div
          className="video-panel clickable"
          onClick={() => setMaximizedPanel('enhanced')}
        >
          <div className="video-header">
            <h3 className="video-title">Enhanced Feed</h3>
            <div className="video-header-controls">
              <span className="badge-live">AI ENHANCED</span>
              {/* Phase-3: Snapshot Capture Button */}
              <button
                className="capture-btn"
                onClick={handleCaptureSnapshot}
                title="Capture Snapshot"
              >
                📸 Capture
              </button>
              <button
                className="expand-btn"
                onClick={(e) => { e.stopPropagation(); setMaximizedPanel('enhanced'); }}
                title="Expand"
              >⛶</button>
            </div>
          </div>
          <div className="video-content">
            <img
              src={displayFrame.image_data || "https://placehold.co/640x480/0A0A0A/737373?text=Awaiting+Signal"}
              alt="Enhanced Feed"
              className="video-feed"
            />
            <div className="maximize-hint">Click to expand</div>

            {/* Detection Overlay with enhanced visuals */}
            {displayFrame.detections && (
              <DetectionOverlay
                detections={displayFrame.detections}
                systemState={displayFrame.state}
                width={640}
                height={480}
              />
            )}

            {/* Safe Mode Overlay on Enhanced Feed */}
            <SafeModeOverlay
              isActive={systemStatus.inSafeMode}
              message={systemStatus.message}
              cause={systemStatus.cause}
            />
          </div>
        </div>

        {/* Alert Panel - Moved inside main-content to span columns */}
        <div className="alert-panel-wrapper">
          <AlertPanel
            currentState={displayFrame.state}
            detections={displayFrame.detections}
            maxConfidence={displayFrame.max_confidence}
            addEvent={addEvent}
          />
        </div>

        {/* Connection Overlay - non-blocking, canvas stays mounted */}
        <ConnectionOverlay
          connectionStatus={connectionStatus}
          reconnectAttempt={reconnectAttempt}
          onRetry={manualReconnect}
        />
      </main>

      {/* Maximized Panel Modal - Raw Feed */}
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

      {/* Maximized Panel Modal - Enhanced Feed */}
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

      {/* Phase-3: Snapshot Modal */}
      <SnapshotModal
        isOpen={snapshotModal.isOpen}
        onClose={closeSnapshotModal}
        imageData={snapshotModal.imageData}
        timestamp={snapshotModal.timestamp}
        alertType={snapshotModal.alertType}
      />
    </div>
  );
}

export default App;

