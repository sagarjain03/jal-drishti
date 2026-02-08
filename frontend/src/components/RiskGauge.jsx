import React from 'react';
import { FUSION_STATE_COLORS, FUSION_STATE_LABELS } from '../constants';
import '../App.css';

/**
 * RiskGauge Component (MILESTONE-3)
 * 
 * Visual representation of the composite risk score with:
 * - Circular gauge showing overall risk level
 * - Individual sensor contribution bars
 * - Color-coded risk zones
 * 
 * Key principle: Shows operators HOW risk is computed
 */
const RiskGauge = ({
    riskScore = 0,
    sensorContributions = { sonar: 0, ir: 0, camera: 0 },
    fusionState = 'NORMAL'
}) => {
    // Convert risk score to percentage
    const riskPercent = Math.round(riskScore * 100);

    // Determine risk zone color
    const getRiskColor = () => {
        if (riskPercent < 15) return '#22C55E'; // Green
        if (riskPercent < 35) return '#3B82F6'; // Blue
        if (riskPercent < 60) return '#F97316'; // Amber
        return '#EF4444'; // Red
    };

    // Get risk zone label
    const getRiskLabel = () => {
        if (riskPercent < 15) return 'LOW';
        if (riskPercent < 35) return 'ELEVATED';
        if (riskPercent < 60) return 'HIGH';
        return 'CRITICAL';
    };

    // Calculate stroke dashoffset for circular gauge
    const circumference = 2 * Math.PI * 45; // radius = 45
    const strokeDashoffset = circumference - (riskPercent / 100) * circumference;

    return (
        <div className="risk-gauge-container">
            {/* Header */}
            <div className="risk-gauge-header">
                <span className="gauge-icon">⚠️</span>
                <span className="gauge-title">RISK SCORE</span>
                <span
                    className="risk-label"
                    style={{ color: getRiskColor() }}
                >
                    {getRiskLabel()}
                </span>
            </div>

            {/* Circular Gauge */}
            <div className="gauge-visual">
                <svg className="risk-ring" viewBox="0 0 100 100">
                    {/* Background ring */}
                    <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#262626"
                        strokeWidth="8"
                    />
                    {/* Progress ring */}
                    <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke={getRiskColor()}
                        strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                        transform="rotate(-90 50 50)"
                        style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.3s ease' }}
                    />
                </svg>
                <div className="gauge-center">
                    <span className="gauge-value" style={{ color: getRiskColor() }}>
                        {riskPercent}
                    </span>
                    <span className="gauge-unit">%</span>
                </div>
            </div>

            {/* Sensor Contributions */}
            <div className="contributions-section">
                <div className="contributions-title">Sensor Contributions</div>

                {/* Sonar */}
                <div className="contribution-row">
                    <span className="contrib-label">🔊 Sonar</span>
                    <div className="contrib-bar-container">
                        <div
                            className="contrib-bar sonar-bar"
                            style={{ width: `${Math.min(100, sensorContributions.sonar * 500)}%` }}
                        />
                    </div>
                    <span className="contrib-value">{(sensorContributions.sonar * 100).toFixed(0)}%</span>
                </div>

                {/* IR */}
                <div className="contribution-row">
                    <span className="contrib-label">🌡️ IR</span>
                    <div className="contrib-bar-container">
                        <div
                            className="contrib-bar ir-bar"
                            style={{ width: `${Math.min(100, sensorContributions.ir * 333)}%` }}
                        />
                    </div>
                    <span className="contrib-value">{(sensorContributions.ir * 100).toFixed(0)}%</span>
                </div>

                {/* Camera */}
                <div className="contribution-row">
                    <span className="contrib-label">📷 Cam</span>
                    <div className="contrib-bar-container">
                        <div
                            className="contrib-bar camera-bar"
                            style={{ width: `${Math.min(100, sensorContributions.camera * 200)}%` }}
                        />
                    </div>
                    <span className="contrib-value">{(sensorContributions.camera * 100).toFixed(0)}%</span>
                </div>
            </div>

            {/* Footer Note */}
            <div className="gauge-footer">
                <span className="footer-note">Risk = Weighted Sensor Fusion</span>
            </div>
        </div>
    );
};

export default RiskGauge;
