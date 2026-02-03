import React, { useEffect, useCallback } from 'react';
import '../App.css';

/**
 * MaximizedPanel Component
 * 
 * Fullscreen modal for displaying video panels in expanded view.
 * Features:
 * - Dark blur backdrop
 * - Close on ESC key or X button
 * - Smooth fade/scale animation
 * - Maintains live stream updates
 */
const MaximizedPanel = ({ isOpen, onClose, title, badge, children }) => {
    // Handle ESC key to close
    const handleKeyDown = useCallback((e) => {
        if (e.key === 'Escape') {
            onClose();
        }
    }, [onClose]);

    // Handle backdrop click to close
    const handleBackdropClick = (e) => {
        if (e.target.classList.contains('maximized-backdrop')) {
            onClose();
        }
    };

    // Add/remove ESC key listener
    useEffect(() => {
        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';
        }
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [isOpen, handleKeyDown]);

    if (!isOpen) return null;

    return (
        <div
            className="maximized-backdrop"
            onClick={handleBackdropClick}
        >
            <div className="maximized-panel">
                {/* Header with title and close button */}
                <div className="maximized-header">
                    <div className="maximized-title-group">
                        <h2 className="maximized-title">{title}</h2>
                        {badge && (
                            <span className="maximized-badge">{badge}</span>
                        )}
                    </div>
                    <button
                        className="maximized-close-btn"
                        onClick={onClose}
                        aria-label="Close fullscreen"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>

                {/* Content area - receives video feed */}
                <div className="maximized-content">
                    {children}
                </div>

                {/* Footer hint */}
                <div className="maximized-footer">
                    <span className="maximized-hint">Press ESC or click outside to close</span>
                </div>
            </div>
        </div>
    );
};

export default MaximizedPanel;
