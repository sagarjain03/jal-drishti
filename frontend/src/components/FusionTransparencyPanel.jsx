/**
 * FusionTransparencyPanel.jsx
 * 
 * Displays the 6-step confidence pipeline breakdown.
 * Makes the system explainable — critical for defense-style UI.
 * 
 * Shows:
 *   ML: 0.62 | Stability: ×0.81 | Persistence: ×0.76
 *   Health: ×0.92 | Correlation: +0.08 | Final: 0.48
 */

import React from 'react';
import { SAFE_MODE_REASONS } from '../system/systemConfig';

const FusionTransparencyPanel = ({
  breakdown = {},
  isVolatile = false,
  lowSignalReliability = false,
  safeModeReason = null,
  similarObjectCount = 0
}) => {
  const {
    step1_mlBase = 0,
    step2_stability = 1,
    step3_persistence = 1,
    step4_healthScaling = 1,
    step5_correlationBoost = 0,
    step5_noisePenalty = 1,
    step6_final = 0.05
  } = breakdown;

  const finalColor = step6_final >= 0.75 ? '#EF4444'
    : step6_final >= 0.40 ? '#F97316'
    : '#22C55E';

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(15,15,25,0.95), rgba(20,20,35,0.9))',
      border: '1px solid rgba(100, 140, 255, 0.15)',
      borderRadius: '10px',
      padding: '12px',
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      fontSize: '11px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <span style={{ color: '#8B9FFF', fontWeight: 700, fontSize: '11px', letterSpacing: '0.5px' }}>
          FUSION PIPELINE
        </span>
        <div style={{ display: 'flex', gap: '4px' }}>
          {lowSignalReliability && (
            <span style={{ background: '#F97316', color: '#000', padding: '1px 6px', borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>
              LOW SIGNAL
            </span>
          )}
          {isVolatile && (
            <span style={{ background: '#EF4444', color: '#fff', padding: '1px 6px', borderRadius: '3px', fontSize: '9px', fontWeight: 700 }}>
              VOLATILE
            </span>
          )}
        </div>
      </div>

      {/* Pipeline Steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <PipelineRow label="ML Base" value={step1_mlBase} prefix="" />
        <PipelineRow label="Stability" value={step2_stability} prefix="×" />
        <PipelineRow label="Persistence" value={step3_persistence} prefix="×" />
        <PipelineRow label="Health" value={step4_healthScaling} prefix="×" />
        <PipelineRow label="Correlation" value={step5_correlationBoost} prefix="+" />
        {step5_noisePenalty < 1 && (
          <PipelineRow label="Noise Penalty" value={step5_noisePenalty} prefix="×" warn />
        )}
        <div style={{ borderTop: '1px solid rgba(100,140,255,0.2)', margin: '2px 0' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: '#CBD5E1', fontWeight: 700 }}>FINAL</span>
          <span style={{ color: finalColor, fontWeight: 700, fontSize: '14px' }}>
            {step6_final.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Object Count + Reason */}
      <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between', color: '#64748B', fontSize: '10px' }}>
        <span>Objects: {similarObjectCount}</span>
        {safeModeReason && safeModeReason !== SAFE_MODE_REASONS.INITIAL && (
          <span style={{ color: '#F97316' }}>
            {safeModeReason.replace(/_/g, ' ')}
          </span>
        )}
      </div>
    </div>
  );
};

const PipelineRow = ({ label, value, prefix, warn = false }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1px 0' }}>
    <span style={{ color: warn ? '#F97316' : '#94A3B8' }}>{label}</span>
    <span style={{ color: warn ? '#F97316' : '#CBD5E1' }}>
      {prefix}{typeof value === 'number' ? value.toFixed(2) : value}
    </span>
  </div>
);

export default FusionTransparencyPanel;
