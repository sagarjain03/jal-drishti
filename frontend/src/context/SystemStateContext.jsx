/**
 * SystemStateContext.jsx
 * 
 * React Context wrapping SystemStateManager.
 * Provides globalSystemState, fusionBreakdown, alerts, and sensor data to all components.
 * 
 * Architecture:
 *   StreamContext (WebSocket) → SystemStateManager → This Context → UI
 *   SonarEngine → SystemStateManager (via context methods)
 *   InfraredEngine → SystemStateManager (via context methods)
 */

import React, { createContext, useContext, useState, useRef, useCallback, useEffect } from 'react';
import { SystemStateManager } from '../system/systemStateManager';
import { useStream } from './StreamContext';

const SystemStateContext = createContext(null);

/**
 * Provider that wraps the app and provides centralized system state.
 */
export const SystemStateProvider = ({ children }) => {
  const { frame, systemStatus } = useStream();
  const managerRef = useRef(new SystemStateManager());
  const [globalState, setGlobalState] = useState(managerRef.current.getState());

  // Process each new frame through the state manager
  useEffect(() => {
    if (frame) {
      const newState = managerRef.current.processFrame(frame);
      setGlobalState({ ...newState });
    }
  }, [frame]);

  /**
   * Called by useSonarEngine() to feed raw sonar data.
   */
  const updateSonarData = useCallback((sonarData) => {
    managerRef.current.updateSonarData(sonarData);
  }, []);

  /**
   * Called by useInfraredEngine() to feed raw IR data.
   */
  const updateIRData = useCallback((irData) => {
    managerRef.current.updateIRData(irData);
  }, []);

  const value = {
    // Core state — all UI reads from here
    globalState,

    // Breakdown for Fusion Transparency Panel
    pipelineBreakdown: globalState.pipelineBreakdown,

    // Alerts
    alerts: globalState.alerts,

    // Temporal buffer
    temporalBuffer: globalState.temporalBuffer,
    similarObjectCount: globalState.similarObjectCount,

    // Correlation
    correlation: globalState.correlation,

    // Flags
    lowSignalReliability: globalState.lowSignalReliability,
    anySensorOffline: globalState.anySensorOffline,
    fusionStability: globalState.fusionStability,
    isVolatile: globalState.isVolatile,
    safeModeReason: globalState.safeModeReason,

    // System status passthrough (safe mode overlay from backend)
    systemStatus,

    // Sensor data feed methods
    updateSonarData,
    updateIRData
  };

  return (
    <SystemStateContext.Provider value={value}>
      {children}
    </SystemStateContext.Provider>
  );
};

/**
 * Hook to consume SystemState context.
 */
export const useSystemState = () => {
  const context = useContext(SystemStateContext);
  if (!context) {
    throw new Error('useSystemState must be used within a SystemStateProvider');
  }
  return context;
};

export default SystemStateContext;
