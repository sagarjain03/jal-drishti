/**
 * infraredEngine.js
 * 
 * Orchestrator hook: useInfraredEngine()
 * 
 * Produces RAW IR data only — NO fusion logic.
 * Feeds raw data to SystemStateManager via context.
 * 
 * Returns: live zones, metrics, heatmap, stability, material, drift, area, confidence trend.
 */

import { useState, useEffect, useRef } from 'react';
import { ThermalDataGenerator } from './thermalDataGenerator';
import {
  computeStabilityIndex,
  classifyMaterial,
  computeDriftRate,
  computeAreaM2,
  computeZoneLevel,
  computeIRConfidence
} from './thermalMath';
import { IR_TICK_INTERVAL_MS, IR_CONFIDENCE_BUFFER_SIZE, BACKGROUND_TEMP } from './infraredConfig';

/**
 * Custom React hook for the infrared sensor engine.
 * 
 * @param {function|null} onUpdate - Callback to feed raw data to SystemStateManager
 * @returns {object} Live IR state
 */
export const useInfraredEngine = (onUpdate = null) => {
  const generatorRef = useRef(new ThermalDataGenerator());
  const confidenceTrendRef = useRef([]);

  const [irState, setIRState] = useState(() => buildInitialState());

  useEffect(() => {
    const interval = setInterval(() => {
      const gen = generatorRef.current;
      const tickData = gen.tick();

      // Process each zone
      const processedZones = tickData.zones.map(zone => {
        // Stability
        const tempValues = zone.tempHistory.map(h => h.value);
        const stability = computeStabilityIndex(tempValues);

        // Material classification
        const material = classifyMaterial({
          thermalInertia: zone.thermalInertia,
          diffusionRate: zone.diffusionRate,
          shapeCoherence: zone.shapeCoherence
        });

        // Drift rate
        const driftResult = computeDriftRate(zone.tempHistory);

        // Area
        const area = computeAreaM2(zone);

        // Zone level
        const level = computeZoneLevel(zone.heatDelta);

        return {
          ...zone,
          stability,
          material,
          driftRate: driftResult.driftRate,
          driftTrend: driftResult.trend,
          area,
          level
        };
      });

      // Overall IR confidence (from hottest zone)
      const hottestZone = processedZones.reduce((best, z) =>
        (!best || z.heatDelta > best.heatDelta) ? z : best, null);

      const irConfidence = hottestZone
        ? computeIRConfidence(hottestZone.heatDelta, hottestZone.stability)
        : 0;

      // Track confidence trend
      const now = Date.now();
      confidenceTrendRef.current.push({ time: now, value: irConfidence });
      if (confidenceTrendRef.current.length > IR_CONFIDENCE_BUFFER_SIZE) {
        confidenceTrendRef.current.shift();
      }

      // Overall stability (average across zones)
      const avgStability = processedZones.length > 0
        ? Math.round(
            processedZones.reduce((sum, z) => sum + z.stability, 0) / processedZones.length * 100
          ) / 100
        : 1.0;

      // Thermal stability label
      let thermalStability;
      if (avgStability >= 0.8) thermalStability = 'CONSISTENT';
      else if (avgStability >= 0.5) thermalStability = 'MODERATE';
      else thermalStability = 'UNSTABLE';

      // Build state
      const newState = {
        // Zones
        zones: processedZones,

        // Heatmap
        heatmapGrid: tickData.heatmapGrid,

        // Overall metrics
        heatDelta: tickData.heatDelta,
        backgroundTemp: tickData.backgroundTemp,
        irConfidence,
        sensorHealth: tickData.sensorHealth,

        // Stability
        avgStability,
        thermalStability,

        // Material from hottest zone
        signatureType: hottestZone?.material?.type || 'AMBIENT',
        materialBreakdown: hottestZone?.material?.breakdown || { thermalInertia: 0, diffusionRate: 0, shapeCoherence: 0 },

        // Confidence trend
        confidenceTrend: [...confidenceTrendRef.current],

        // Per-zone summaries for panels
        zoneSummaries: processedZones.map(z => ({
          id: z.id,
          label: z.label,
          temp: z.currentTemp,
          heatDelta: z.heatDelta,
          stability: z.stability,
          material: z.material.type,
          driftRate: z.driftRate,
          driftTrend: z.driftTrend,
          area: z.area,
          level: z.level
        }))
      };

      setIRState(newState);

      if (onUpdate) {
        onUpdate(newState);
      }
    }, IR_TICK_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [onUpdate]);

  return irState;
};

function buildInitialState() {
  return {
    zones: [],
    heatmapGrid: [],
    heatDelta: 0,
    backgroundTemp: BACKGROUND_TEMP,
    irConfidence: 0,
    sensorHealth: 0.92,
    avgStability: 1.0,
    thermalStability: 'CONSISTENT',
    signatureType: 'AMBIENT',
    materialBreakdown: { thermalInertia: 0, diffusionRate: 0, shapeCoherence: 0 },
    confidenceTrend: [],
    zoneSummaries: []
  };
}

export default useInfraredEngine;
