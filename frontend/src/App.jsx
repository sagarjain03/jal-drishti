import React, { useRef, useEffect, useState } from 'react';
import StatusBar from './components/StatusBar';
import VideoPanel from './components/VideoPanel';
import AlertPanel from './components/AlertPanel';
import ConnectionOverlay from './components/ConnectionOverlay';
import LoginPage from './components/LoginPage';
import useLiveStream from './hooks/useLiveStream';
import useFakeStream from './hooks/useFakeStream';
import { SYSTEM_STATES, CONNECTION_STATES } from './constants';
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
 * App Component
 * 
 * Main dashboard for Jal-Drishti Phase-2.
 * - Uses useRef for state tracking (avoids render storms)
 * - Displays ConnectionOverlay when disconnected
 * - Passes all required data to child components
 */
function App() {
  const [token, setToken] = useState(null);

  // Only use live stream when not in fake/test mode
  const liveStreamData = useLiveStream(USE_FAKE_STREAM ? null : token);

  // Destructure with defaults for when hook returns null/disconnected state
  const {
    frame = null,
    fps = 0,
    connectionStatus = CONNECTION_STATES.CONNECTED, // When fake, pretend connected
    reconnectAttempt = 0,
    lastValidFrame = null,
    manualReconnect = () => { }
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

  if (!token) {
    return <LoginPage onLogin={setToken} />;
  }

  return (
    <div className="app-container">
      {/* Test Mode Indicator */}
      {USE_FAKE_STREAM && (
        <div className="test-mode-banner">
          🧪 TEST MODE: Using Fake Stream (state cycles every ~10s)
        </div>
      )}

      <StatusBar
        systemState={displayFrame.state}
        maxConfidence={displayFrame.max_confidence}
        latencyMs={displayFrame.system?.latency_ms}
        renderFps={fps}
        mlFps={displayFrame.system?.fps}
        connectionStatus={connectionStatus}
      />

      <main className="main-content">
        <VideoPanel
          title="Raw Feed"
          imageSrc="https://placehold.co/640x480/333/FFF?text=Raw+Feed"
          isEnhanced={false}
        />
        <VideoPanel
          title="Enhanced Feed"
          imageSrc={displayFrame.image_data}
          detections={displayFrame.detections}
          systemState={displayFrame.state}
          isEnhanced={true}
        />

        {/* Connection Overlay - non-blocking, canvas stays mounted */}
        <ConnectionOverlay
          connectionStatus={connectionStatus}
          reconnectAttempt={reconnectAttempt}
          onRetry={manualReconnect}
        />
      </main>

      <AlertPanel
        currentState={displayFrame.state}
        detections={displayFrame.detections}
        maxConfidence={displayFrame.max_confidence}
      />
    </div>
  );
}

export default App;
