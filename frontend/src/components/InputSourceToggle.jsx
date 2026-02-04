import React from 'react';
import '../App.css';

/**
 * InputSourceToggle Component
 * Allows user to switch between Video File and Camera input sources.
 * Fills the empty space in the left sidebar.
 */
const InputSourceToggle = ({ currentSource, onToggle }) => {
    const isCamera = currentSource === 'camera';

    return (
        <div className="input-source-toggle-container">
            <div className="toggle-header">
                <span className="toggle-label">INPUT SOURCE CONTROL</span>
                <span className={`toggle-status ${isCamera ? 'status-camera' : 'status-video'}`}>
                    {isCamera ? 'LIVE CAMERA' : 'VIDEO LOOP'}
                </span>
            </div>

            <div className="toggle-controls">
                <button
                    className={`source-btn ${!isCamera ? 'active' : ''}`}
                    onClick={() => onToggle('video')}
                    title="Use Pre-recorded Video"
                >
                    <span className="source-icon">🎬</span>
                    <div className="source-info">
                        <span className="source-name">VIDEO FILE</span>
                        <span className="source-detail">RTSP / MP4</span>
                    </div>
                </button>

                <div className="toggle-divider"></div>

                <button
                    className={`source-btn ${isCamera ? 'active' : ''}`}
                    onClick={() => onToggle('camera')}
                    title="Use Webcam Feed"
                >
                    <span className="source-icon">📷</span>
                    <div className="source-info">
                        <span className="source-name">WEBCAM</span>
                        <span className="source-detail">USB / INTEGRATED</span>
                    </div>
                </button>
            </div>

            <div className="toggle-footer">
                <div className="signal-strength">
                    <span>SIGNAL:</span>
                    <div className="signal-bars">
                        <div className="bar full"></div>
                        <div className="bar full"></div>
                        <div className="bar full"></div>
                        <div className="bar full"></div>
                    </div>
                </div>
                <span className="source-id">ID: SRC-{isCamera ? 'CAM-01' : 'VID-09'}</span>
            </div>
        </div>
    );
};

export default InputSourceToggle;
