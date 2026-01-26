import React from 'react';
import DetectionOverlay from './DetectionOverlay';
import { SYSTEM_STATES } from '../constants';
import '../App.css';

/**
 * VideoPanel Component
 * 
 * Displays video feed with optional detection overlay.
 * - Raw feed: no overlay
 * - Enhanced feed: with bounding boxes
 * - Passes system state to overlay for proper coloring
 */
const VideoPanel = ({
    title,
    imageSrc,
    detections,
    systemState = SYSTEM_STATES.SAFE_MODE,
    isEnhanced = false
}) => {
    return (
        <div className="video-panel">
            <div className="video-header">
                <h3 className="video-title">{title}</h3>
                {isEnhanced && <span className="badge-live">AI ENHANCED</span>}
            </div>

            <div className="video-content">
                {/* Feed Image */}
                <img
                    src={imageSrc || "https://placehold.co/640x480/111/FFF?text=No+Signal"}
                    alt={title}
                    className="video-feed"
                />

                {/* Detection Overlay - Only on Enhanced feed */}
                {isEnhanced && detections && (
                    <DetectionOverlay
                        detections={detections}
                        systemState={systemState}
                        width={640}
                        height={480}
                    />
                )}
            </div>
        </div>
    );
};

export default VideoPanel;
