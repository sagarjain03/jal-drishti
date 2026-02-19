import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import StatusBar from '../components/StatusBar';
import ConnectionOverlay from '../components/ConnectionOverlay';
import TacticalNav from '../components/TacticalNav';
import '../App.css';

// Import StreamContext consumption
import { useStream } from '../context/StreamContext';

const MainLayout = () => {
    const {
        frame,
        fps,
        connectionStatus,
        reconnectAttempt,
        manualReconnect,
        systemStatus,
        lastValidFrame
    } = useStream();

    const [inputSource, setInputSource] = useState('video');
    const [navCollapsed, setNavCollapsed] = useState(false);

    const location = useLocation();

    // Determine display frame
    const displayFrame = frame || lastValidFrame || {
        state: 'SAFE_MODE',
        max_confidence: 0,
        detections: [],
        image_data: null,
        system: { fps: null, latency_ms: null },
        risk_score: 0
    };

    return (
        <div className={`app-container ${systemStatus.inSafeMode ? 'safe-mode-active' : ''}`}>

            {/* Status Bar - Persistent across pages */}
            <StatusBar
                systemState={displayFrame.state}
                maxConfidence={displayFrame.max_confidence}
                latencyMs={displayFrame.system?.latency_ms}
                renderFps={fps}
                mlFps={displayFrame.system?.fps}
                connectionStatus={connectionStatus}
                inputSource={inputSource}
                uptime="00:00:00"
            />

            <div className="main-content-wrapper" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

                {/* Tactical Navigation Panel */}
                <TacticalNav 
                    connectionStatus={connectionStatus} 
                    collapsed={navCollapsed}
                    onToggle={() => setNavCollapsed(!navCollapsed)}
                />

                {/* Main Page Content */}
                <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                    <Outlet context={{ displayFrame, systemStatus, inputSource, setInputSource, navCollapsed }} />
                </div>
            </div>

            {/* Global Overlays */}
            <ConnectionOverlay
                connectionStatus={connectionStatus}
                reconnectAttempt={reconnectAttempt}
                onRetry={manualReconnect}
            />
        </div>
    );
};

export default MainLayout;
