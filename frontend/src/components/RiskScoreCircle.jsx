
import React from 'react';
import '../App.css';
import { SYSTEM_STATES } from '../constants';

/**
 * RiskScoreCircle Component
 * 
 * Central visual element of the dashboard.
 * Displays the current system risk level as a circular gauge.
 * 
 * - Green: Safe
 * - Amber: Potential Anomaly
 * - Red: Confirmed Threat
 */
const RiskScoreCircle = ({ systemState, confidence, isSafeMode }) => {

    // Determine color and label based on state
    let riskColor = 'var(--accent-green)';
    let riskLabel = 'SAFE';
    let pulseClass = '';

    // Calculate progress (visual representation only)
    // 0-33: Safe, 34-66: Warning, 67-100: Threat
    let progress = 15; // default safe

    if (isSafeMode) {
        riskColor = 'var(--accent-green)';
        riskLabel = 'SAFE MODE';
    } else {
        switch (systemState) {
            case SYSTEM_STATES.CONFIRMED_THREAT:
                riskColor = 'var(--accent-red)';
                riskLabel = 'CRITICAL';
                pulseClass = 'pulse-critical';
                progress = 75 + (confidence * 0.25); // 75-100
                break;
            case SYSTEM_STATES.POTENTIAL_ANOMALY:
                riskColor = 'var(--accent-amber)';
                riskLabel = 'WARNING';
                pulseClass = 'pulse-warning';
                progress = 40 + (confidence * 0.25); // 40-65
                break;
            default:
                riskColor = 'var(--accent-green)';
                riskLabel = 'NORMAL';
                progress = 10;
        }
    }

    // SVG parameters
    const radius = 45; // Fits inside 100px container
    const stroke = 6;
    const normalizedRadius = radius - stroke * 2;
    const circumference = normalizedRadius * 2 * Math.PI;
    const strokeDashoffset = circumference - (progress / 100) * circumference;


    return (
        <div className={`risk-circle-container ${pulseClass}`}>
            <div className="risk-content">
                <span className="risk-label">RISK LEVEL</span>
                <span className="risk-value" style={{ color: riskColor }}>
                    {riskLabel}
                </span>
                {confidence > 0 && (
                    <span className="risk-confidence">
                        {Math.round(confidence * 100)}% CONFIDENCE
                    </span>
                )}
            </div>

            {/* SVG Ring */}
            <svg
                height={radius * 2}
                width={radius * 2}
                className="risk-ring"
            >
                <circle
                    stroke="var(--bg-elevated)"
                    strokeWidth={stroke}
                    fill="transparent"
                    r={normalizedRadius}
                    cx={radius}
                    cy={radius}
                />
                <circle
                    stroke={riskColor}
                    strokeDasharray={circumference + ' ' + circumference}
                    style={{ strokeDashoffset }}
                    strokeWidth={stroke}
                    strokeLinecap="round"
                    fill="transparent"
                    r={normalizedRadius}
                    cx={radius}
                    cy={radius}
                    className="risk-progress"
                />
            </svg>
        </div>
    );
};

export default RiskScoreCircle;
