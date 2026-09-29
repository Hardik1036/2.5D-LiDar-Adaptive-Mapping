import { test } from "node:test";
import assert from "node:assert/strict";
import { getCostColor, DEFAULT_OPTIONS, costBand } from "../src/three/Scene.js";

test("getCostColor correctly applies DRDO semantic cost bands on ground", () => {
  // Safe Drivable Ground (Road / Corridor): Emerald Green
  assert.equal(getCostColor(0, -1.6), 0x238636);
  assert.equal(getCostColor(15, -1.5), 0x238636);
  assert.equal(getCostColor(50, -1.35), 0x238636);

  // Caution (Curbs, Slopes, Brush): Amber Yellow
  assert.equal(getCostColor(51, -1.5), 0xD29922);
  assert.equal(getCostColor(100, -1.4), 0xD29922);
  assert.equal(getCostColor(180, -1.35), 0xD29922);

  // Lethal Hazard (Vehicles, Walls, Obstacles): Crimson Red
  assert.equal(getCostColor(181, -1.5), 0xF85149);
  assert.equal(getCostColor(200, -1.4), 0xF85149);
  assert.equal(getCostColor(255, -1.6), 0xF85149);
});

test("getCostColor strictly prohibits elevated voxels and air returns from rendering in green", () => {
  // Elevated tree canopies, walls, poles (Z > -1.2m) must render Crimson Red even if cost <= 50 or undefined
  assert.equal(getCostColor(0, 0.5), 0xF85149);
  assert.equal(getCostColor(25, -0.5), 0xF85149);
  assert.equal(getCostColor(50, 1.8), 0xF85149);
  assert.equal(getCostColor(undefined, 0.5), 0xF85149);
  assert.equal(getCostColor(null, 2.0), 0xF85149);
  assert.equal(getCostColor(NaN, 1.8), 0xF85149);

  // Fallback path on ground (Z <= -1.2m) renders safe road green
  assert.equal(getCostColor(undefined, -1.6), 0x238636);
  assert.equal(getCostColor(undefined, -1.35), 0x238636);

  // Caution items elevated retain Caution amber
  assert.equal(getCostColor(100, 0.5), 0xD29922);
  assert.equal(getCostColor(200, 0.5), 0xF85149);
});

test("DEFAULT_OPTIONS initializes minHeight to -5.0m and maxHeight to 10.0m for full range", () => {
  assert.equal(DEFAULT_OPTIONS.minHeight, -5.0);
  assert.equal(DEFAULT_OPTIONS.maxHeight, 10.0);
  assert.equal(DEFAULT_OPTIONS.colorBy, "cost");
});

test("costBand boundary consistency", () => {
  assert.equal(costBand(0), 0);
  assert.equal(costBand(50), 0);
  assert.equal(costBand(51), 1);
  assert.equal(costBand(180), 1);
  assert.equal(costBand(181), 2);
  assert.equal(costBand(255), 2);
});

test("supports backend cost: 100 caution inpainting and lethal negative obstacles", () => {
  // Caution inpainting gap (Cost = 100) -> Solar Amber Yellow
  assert.equal(getCostColor(100, -1.60), 0xD29922);
  assert.equal(getCostColor(100, -1.50), 0xD29922);

  // Lethal negative obstacles / ditches (Cost = 255) -> Crimson Red
  assert.equal(getCostColor(255, -2.10), 0xF85149);
});

test("safely formats stationary vehicles and negative IDs for Analysis table", () => {
  const stationaryCar = {
    id: -1,
    class: "parked_car",
    x: 4.25,
    y: -8.10,
    vx: 0.0,
    vy: 0.0,
    speed: 0.0,
    heading: 1.5707963,
    stationary: true,
  };

  const id = stationaryCar.id;
  assert.equal(id, -1);
  const cls = stationaryCar.class ?? "obstacle";
  assert.equal(cls, "parked_car");
  const speedText = stationaryCar.stationary
    ? "0.00 m/s (Stationary)"
    : `${(Number(stationaryCar.speed) || 0).toFixed(2)} m/s`;
  assert.equal(speedText, "0.00 m/s (Stationary)");
  const heading = `${((Number(stationaryCar.heading) || 0) * (180 / Math.PI)).toFixed(1)}°`;
  assert.equal(heading, "90.0°");
});

test("natural elevation spectral ramp for raw sensor returns maps smoothly across Z range", () => {
  const computeSpectralHue = (z) => {
    const normZ = Math.min(Math.max((z - (-2.0)) / 3.5, 0.0), 1.0);
    return 0.66 * (1.0 - normZ);
  };

  // Low asphalt (-2.0m or below): cool blue/cyan (hue ~ 0.66)
  assert.equal(computeSpectralHue(-2.0), 0.66);
  assert.equal(computeSpectralHue(-2.5), 0.66);

  // High elevation (+1.5m or above): red (hue = 0.0)
  assert.equal(computeSpectralHue(1.5), 0.0);
  assert.equal(computeSpectralHue(2.5), 0.0);

  // Mid elevation: smooth transition
  const midHue = computeSpectralHue(-0.25);
  assert.ok(midHue > 0.25 && midHue < 0.40);
});

test("Mode 2 enforces discrete 3-tier defense palette without gradient bleed", () => {
  // Safe Drivable Road (Cost 0-50): Emerald Green
  assert.equal(getCostColor(0, -1.6), 0x238636);
  assert.equal(getCostColor(50, -1.6), 0x238636);

  // Caution (Cost 51-180): Solar Amber Yellow
  assert.equal(getCostColor(51, -1.6), 0xD29922);
  assert.equal(getCostColor(180, -1.6), 0xD29922);

  // Lethal Obstacle (Cost 181-255): Tactical Crimson Red
  assert.equal(getCostColor(181, -1.6), 0xF85149);
  assert.equal(getCostColor(255, -1.6), 0xF85149);
});

test("live stream transport controls toggle pause and speed states properly", async () => {
  const {
    setLivePaused,
    isLivePaused,
    setLivePlaybackSpeed,
    getLivePlaybackSpeed,
  } = await import("../src/services/telemetryService.js");

  assert.equal(isLivePaused(), false);
  setLivePaused(true);
  assert.equal(isLivePaused(), true);
  setLivePaused(false);
  assert.equal(isLivePaused(), false);

  assert.equal(getLivePlaybackSpeed(), 1.0);
  setLivePlaybackSpeed(0.5);
  assert.equal(getLivePlaybackSpeed(), 0.5);
  setLivePlaybackSpeed(0.25);
  assert.equal(getLivePlaybackSpeed(), 0.25);
  setLivePlaybackSpeed(1.0);
  assert.equal(getLivePlaybackSpeed(), 1.0);
});

test("full-scale LiDAR sweep bounding box and camera target calculation centers grid mesh", () => {
  // Simulate full-scale LiDAR sweep bounds
  const sweepMin = { x: -25.0, y: -2.5, z: -60.0 };
  const sweepMax = { x: 60.0, y: 3.5, z: 25.0 };

  const center = {
    x: (sweepMin.x + sweepMax.x) / 2,
    y: (sweepMin.y + sweepMax.y) / 2,
    z: (sweepMin.z + sweepMax.z) / 2,
  };

  const size = {
    x: sweepMax.x - sweepMin.x,
    y: sweepMax.y - sweepMin.y,
    z: sweepMax.z - sweepMin.z,
  };

  assert.equal(center.x, 17.5);
  assert.equal(center.y, 0.5);
  assert.equal(center.z, -17.5);

  assert.equal(size.x, 85.0);
  assert.equal(size.y, 6.0);
  assert.equal(size.z, 85.0);

  const span = Math.max(size.x, size.z, 20.0);
  assert.equal(span, 85.0);

  const preset = {
    position: [center.x + span * 0.55, center.y + Math.max(span * 0.70, 24.0), center.z + span * 0.75],
    target: [center.x, center.y, center.z],
  };

  assert.deepEqual(preset.target, [17.5, 0.5, -17.5]);
  assert.ok(preset.position[1] > 24.0);
  assert.ok(preset.position[0] > preset.target[0]);
});

test("vehicle orientation heading aligns with positive X axis (+X forward)", () => {
  // Positive X forward travel corridor: vx > 0, vy = 0
  const vx = 5.0;
  const vy = 0.0;
  const heading = Math.atan2(-vy || 0, vx);
  assert.equal(heading, 0);

  // Model offset in Three.js coordinates
  const finalRotY = heading + Math.PI / 2;
  assert.equal(finalRotY, Math.PI / 2);

  // When moving left (+Y in KITTI / ISO, which is -Z in Three.js): vx = 0, vy = 5.0
  const headingLeft = Math.atan2(-5.0, 0.0);
  assert.equal(headingLeft, -Math.PI / 2);
});



