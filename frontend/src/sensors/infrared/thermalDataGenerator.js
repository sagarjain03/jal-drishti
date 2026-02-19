/**
 * thermalDataGenerator.js
 * 
 * Simulates IR thermal zone data with drift, per-zone temperature history,
 * confidence trend, and heatmap grid data.
 * 
 * Dynamic degradation: low health → more stability fluctuation, noisier temps.
 */

import {
  IR_ZONES,
  BACKGROUND_TEMP,
  HEATMAP_GRID_SIZE,
  DRIFT_RATE_PER_TICK,
  IR_SENSOR_HEALTH_DEFAULT,
  IR_HEALTH_DRIFT,
  IR_HEALTH_STABILITY_AMPLIFICATION,
  IR_HEALTH_TEMP_NOISE
} from './infraredConfig';

export class ThermalDataGenerator {
  constructor() {
    // Per-zone temp history
    this.zones = IR_ZONES.map(z => ({
      ...z,
      currentTemp: z.baseTemp + BACKGROUND_TEMP + (Math.random() * 4 - 1),
      tempHistory: [],
      thermalInertia: 0.5 + Math.random() * 0.4,
      diffusionRate: 0.3 + Math.random() * 0.4,
      shapeCoherence: 0.4 + Math.random() * 0.4
    }));

    // Sensor health
    this.sensorHealth = IR_SENSOR_HEALTH_DEFAULT;

    // Confidence history
    this.confidenceTrend = [];
  }

  /**
   * Advance one tick of simulation.
   */
  tick() {
    const now = Date.now();

    // Slow health drift
    this.sensorHealth = Math.max(0.3, Math.min(1,
      this.sensorHealth + (Math.random() - 0.52) * IR_HEALTH_DRIFT * 2
    ));
    this.sensorHealth = Math.round(this.sensorHealth * 100) / 100;

    // Health-based degradation
    const healthFactor = this.sensorHealth;
    const tempNoise = (1 - healthFactor) * IR_HEALTH_TEMP_NOISE;

    // Update zones
    this.zones = this.zones.map(zone => {
      // Temperature drift
      const drift = (Math.random() - 0.5) * DRIFT_RATE_PER_TICK;
      const healthNoise = (Math.random() - 0.5) * tempNoise;
      let newTemp = zone.currentTemp + drift + healthNoise;
      // Keep near base temp ± variation
      const targetTemp = zone.baseTemp + BACKGROUND_TEMP;
      newTemp = newTemp * 0.98 + targetTemp * 0.02; // slow pull toward baseline
      newTemp = Math.round(newTemp * 10) / 10;

      // Record history
      const newHistory = [...zone.tempHistory, { time: now, value: newTemp }].slice(-30);

      // Slight fluctuation in material properties
      const stabilityFluctuation = (1 - healthFactor) * IR_HEALTH_STABILITY_AMPLIFICATION;

      return {
        ...zone,
        currentTemp: newTemp,
        tempHistory: newHistory,
        thermalInertia: Math.max(0.1, Math.min(1,
          zone.thermalInertia + (Math.random() - 0.5) * 0.02 + (Math.random() - 0.5) * stabilityFluctuation
        )),
        diffusionRate: Math.max(0.1, Math.min(1,
          zone.diffusionRate + (Math.random() - 0.5) * 0.015
        )),
        shapeCoherence: Math.max(0.1, Math.min(1,
          zone.shapeCoherence + (Math.random() - 0.5) * 0.01
        ))
      };
    });

    // Generate heatmap grid
    const heatmapGrid = this._generateHeatmapGrid();

    // Compute overall heat delta
    const maxTemp = Math.max(...this.zones.map(z => z.currentTemp));
    const heatDelta = Math.round((maxTemp - BACKGROUND_TEMP) * 10) / 10;

    return {
      zones: this.zones.map(z => ({
        id: z.id,
        label: z.label,
        currentTemp: z.currentTemp,
        heatDelta: Math.round((z.currentTemp - BACKGROUND_TEMP) * 10) / 10,
        tempHistory: z.tempHistory,
        thermalInertia: Math.round(z.thermalInertia * 100) / 100,
        diffusionRate: Math.round(z.diffusionRate * 100) / 100,
        shapeCoherence: Math.round(z.shapeCoherence * 100) / 100,
        x: z.x,
        y: z.y,
        radius: z.radius
      })),
      heatmapGrid,
      heatDelta,
      backgroundTemp: BACKGROUND_TEMP,
      sensorHealth: this.sensorHealth
    };
  }

  /**
   * Generate heatmap grid data.
   */
  _generateHeatmapGrid() {
    const grid = [];
    for (let y = 0; y < HEATMAP_GRID_SIZE; y++) {
      for (let x = 0; x < HEATMAP_GRID_SIZE; x++) {
        let intensity = 0;

        // Accumulate intensity from each zone
        for (const zone of this.zones) {
          const dx = x - zone.x;
          const dy = y - zone.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < zone.radius * 2) {
            const zoneIntensity = Math.max(0, 1 - dist / (zone.radius * 2));
            const tempFactor = (zone.currentTemp - BACKGROUND_TEMP) / 20;
            intensity += zoneIntensity * Math.max(0, tempFactor);
          }
        }

        grid.push({
          x,
          y,
          intensity: Math.min(1, Math.max(0, intensity))
        });
      }
    }
    return grid;
  }

  getState() {
    return {
      zones: this.zones,
      sensorHealth: this.sensorHealth
    };
  }
}

export default ThermalDataGenerator;
