import React from 'react';
import '../App.css';

/**
 * SnapshotModal Component
 * 
 * Modal for displaying captured frame with timestamp and alert overlays.
 * Features:
 * - Freeze-frame preview
 * - Timestamp overlay
 * - Alert type label
 * - Download functionality
 */
const SnapshotModal = ({ isOpen, onClose, imageData, timestamp, alertType }) => {
    if (!isOpen || !imageData) return null;

    const getAlertClass = () => {
        switch (alertType) {
            case 'CONFIRMED_THREAT': return 'threat';
            case 'POTENTIAL_ANOMALY': return 'anomaly';
            default: return 'normal';
        }
    };

    const getAlertLabel = () => {
        switch (alertType) {
            case 'CONFIRMED_THREAT': return 'THREAT';
            case 'POTENTIAL_ANOMALY': return 'ANOMALY';
            default: return 'NORMAL';
        }
    };

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = imageData;
        link.download = `jal-drishti-snapshot-${timestamp.replace(/[:/]/g, '-')}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleBackdropClick = (e) => {
        if (e.target.classList.contains('snapshot-modal-backdrop')) {
            onClose();
        }
    };

    return (
        <div className="snapshot-modal-backdrop" onClick={handleBackdropClick}>
            <div className="snapshot-modal">
                <div className="snapshot-modal-header">
                    <span className="snapshot-modal-title">
                        📸 Captured Snapshot
                    </span>
                    <button className="snapshot-modal-close" onClick={onClose}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>

                <div className="snapshot-modal-content">
                    <div className="snapshot-image-container">
                        <img src={imageData} alt="Captured Snapshot" className="snapshot-image" />
                        <div className="snapshot-overlay">
                            <span className="snapshot-timestamp">{timestamp}</span>
                            <span className={`snapshot-alert-label ${getAlertClass()}`}>
                                {getAlertLabel()}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="snapshot-modal-footer">
                    <button className="snapshot-download-btn" onClick={handleDownload}>
                        ⬇ Download
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SnapshotModal;
