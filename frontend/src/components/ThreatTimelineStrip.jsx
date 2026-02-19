import React, { useEffect, useState, useRef } from 'react';
import { useSystemState } from '../context/SystemStateContext';

const ThreatTimelineStrip = () => {
    const { globalState } = useSystemState();
    const [history, setHistory] = useState([]);
    const maxItems = 60; // 30 seconds @ 2 updates/sec approx

    useEffect(() => {
        // Add current state to history
        const currentState = globalState?.state || 'SAFE_MODE';
        const timestamp = new Date();
        
        setHistory(prev => {
            const next = [...prev, { state: currentState, time: timestamp }];
            if (next.length > maxItems) return next.slice(next.length - maxItems);
            return next;
        });
    }, [globalState?.state]);

    const getColor = (state) => {
        switch (state) {
            case 'CONFIRMED_THREAT': return '#EF4444'; // Red
            case 'POTENTIAL_THREAT': return '#F97316'; // Amber
            case 'SUSPICIOUS_ACTIVITY': return '#EAB308'; // Yellow
            case 'SAFE_MODE': return '#22C55E'; // Green
            default: return '#333';
        }
    };

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            background: '#0a0a0a',
            border: '1px solid #1A1A1A',
            borderRadius: '6px',
            padding: '8px 12px',
            gap: '4px',
            marginTop: '0px',
            marginBottom: '4px'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '10px', fontWeight: 700, color: '#666', letterSpacing: '1px' }}>THREAT TIMELINE (30s)</span>
                <span style={{ fontSize: '9px', color: '#444', fontFamily: 'monospace' }}>LIVE FEED</span>
            </div>
            
            <div style={{ 
                display: 'flex', 
                height: '12px', 
                gap: '2px', 
                alignItems: 'center' 
            }}>
                {/* Render history blocks */}
                {Array.from({ length: maxItems }).map((_, i) => {
                    const dataIndex = i - (maxItems - history.length);
                    const item = dataIndex >= 0 ? history[dataIndex] : null;
                    
                    return (
                        <div key={i} style={{
                            flex: 1,
                            height: item ? '100%' : '20%',
                            background: item ? getColor(item.state) : '#1A1A1A',
                            borderRadius: '1px',
                            opacity: item ? 1 : 0.3,
                            transition: 'height 0.2s ease, background 0.2s ease'
                        }} title={item ? `${item.state} @ ${item.time.toLocaleTimeString()}` : ''} />
                    );
                })}
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#444' }}>
                <span>-30s</span>
                <span>NOW</span>
            </div>
        </div>
    );
};

export default ThreatTimelineStrip;
