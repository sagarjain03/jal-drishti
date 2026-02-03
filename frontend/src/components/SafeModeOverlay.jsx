import React from 'react';
import '../App.css';

/**
 * SafeModeOverlay Component
 * 
 * Displays a critical visual overlay when system enters safe mode.
 * Features:
 * - Semi-transparent RED overlay on video area
 * - Large warning text with ⚠ icon
 * - Cause/message from backend
 * - Subtle pulsing animation (CSS-only, performant)
 * 
 * IMPORTANT: This is purely visual - does NOT modify backend behavior.
 * Overlay appears on status = "safe_mode" and disappears on status = "recovered"
 */
const SafeModeOverlay = ({ isActive, message, cause }) => {
    if (!isActive) {
        return null;
    }

    return (
        <div className="safe-mode-overlay">
            <div className="safe-mode-content">
                <div className="safe-mode-icon">⚠️</div>
                <h2 className="safe-mode-title">SYSTEM IN SAFE MODE</h2>
                <p className="safe-mode-message">
                    {message || 'System has entered protective safe mode. Operations limited.'}
                </p>
                {cause && (
                    <div className="safe-mode-cause">
                        <div className="safe-mode-cause-label">Cause</div>
                        <div className="safe-mode-cause-text">{cause}</div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SafeModeOverlay;
