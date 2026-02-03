import React, { useRef, useEffect } from 'react';
import { EVENT_TYPES, EVENT_ICONS } from '../constants';
import '../App.css';

/**
 * EventTimeline Component (Enhanced)
 * 
 * A scrolling event log panel with:
 * - Color-coded entries (Red/Green/Blue)
 * - Auto-scroll to latest events
 * - Hover-to-expand event details
 * - Smooth slide-in animations
 * 
 * Format: 🕒 14:02:11 — Event description
 */
const EventTimeline = ({ events = [] }) => {
    const listRef = useRef(null);
    const prevEventCountRef = useRef(0);

    /**
     * Auto-scroll to top when new events arrive
     */
    useEffect(() => {
        if (events.length > prevEventCountRef.current && listRef.current) {
            listRef.current.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
        prevEventCountRef.current = events.length;
    }, [events.length]);

    /**
     * Get icon for event type
     */
    const getEventIcon = (type) => {
        return EVENT_ICONS[type] || '📋';
    };

    /**
     * Get CSS class for event severity
     */
    const getEventClass = (severity) => {
        switch (severity) {
            case 'danger':
                return 'event-danger';
            case 'warning':
                return 'event-warning';
            case 'success':
                return 'event-success';
            case 'info':
            default:
                return 'event-info';
        }
    };

    /**
     * Get additional details text based on event type
     */
    const getEventDetails = (event) => {
        switch (event.type) {
            case EVENT_TYPES.SAFE_MODE_ENTRY:
                return 'System entered degraded mode due to low visibility or detection issues.';
            case EVENT_TYPES.SAFE_MODE_EXIT:
                return 'System recovered and resumed normal operation.';
            case EVENT_TYPES.DETECTION:
                return 'Object detected in surveillance feed.';
            case EVENT_TYPES.STATE_CHANGE:
                return 'System threat level changed.';
            case EVENT_TYPES.CONNECTION:
                return 'WebSocket connection established.';
            case EVENT_TYPES.DISCONNECTION:
                return 'Connection interrupted. Attempting reconnection.';
            default:
                return null;
        }
    };

    return (
        <div className="event-timeline">
            <div className="timeline-header">
                <h3 className="timeline-title">
                    <span className="timeline-title-icon">📊</span>
                    Event Log
                </h3>
                <span className="timeline-count">{events.length} events</span>
            </div>

            <div className="timeline-list" ref={listRef}>
                {events.length === 0 ? (
                    <div className="timeline-empty">
                        No events recorded yet.
                    </div>
                ) : (
                    events.map((event, index) => {
                        const details = getEventDetails(event);
                        return (
                            <div
                                key={event.id}
                                className={`timeline-event ${getEventClass(event.severity)} ${index === 0 ? 'event-latest' : ''}`}
                            >
                                <span className="event-icon">
                                    {getEventIcon(event.type)}
                                </span>
                                <div className="event-content">
                                    <div className="event-time">
                                        🕒 {event.timestamp}
                                    </div>
                                    <div className="event-message">
                                        {event.message}
                                    </div>
                                    {/* Hover-to-expand details */}
                                    {details && (
                                        <div className="event-details">
                                            <div className="event-details-text">
                                                {details}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

export default EventTimeline;

