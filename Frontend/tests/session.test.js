import { test } from "node:test";
import assert from "node:assert/strict";
import {
  summarizeFrame,
  validateRecording,
  MAX_HISTORY,
  MAX_RECORDING_FRAMES,
  MAX_RECORDING_BYTES,
} from "../src/utils/session.js";
const frame = () => ({
  timestamp: 1,
  frame_id: 1,
  system_status: "SIMULATED",
  system_stats: { fps: 25, latency_ms: 1, active_cells: 6, ram_mb: 0.1 },
  cells: [0, 50, 51, 180, 181, 255].map((cost) => ({
    x: 0,
    y: 0,
    size: 1,
    cost,
    z_min: 0,
    z_max: 1,
  })),
  dynamic_objects: [],
});
test("classification bands respect exact boundary values and sum to total", () => {
  const s = summarizeFrame(frame());
  assert.deepEqual(s.bands, [2, 2, 2]);
  assert.equal(
    s.bands.reduce((a, b) => a + b, 0),
    s.total,
  );
  assert.equal(s.stats.active_cells, 6);
});
test("recording contract accepts ordered zero-origin timeline", () => {
  const data = {
    version: 1,
    frames: [
      { elapsed_ms: 0, source: "simulated", frame: frame() },
      { elapsed_ms: 200, source: "live", frame: frame() },
    ],
  };
  assert.equal(validateRecording(data, () => true).length, 2);
});
test("invalid recording versions, timestamps and provenance rejected", () => {
  assert.throws(() =>
    validateRecording({ version: 2, frames: [] }, () => true),
  );
  assert.throws(() =>
    validateRecording(
      {
        version: 1,
        frames: [{ elapsed_ms: -1, source: "simulated", frame: frame() }],
      },
      () => true,
    ),
  );
  assert.throws(() =>
    validateRecording(
      {
        version: 1,
        frames: [{ elapsed_ms: 1, source: "unknown", frame: frame() }],
      },
      () => true,
    ),
  );
});
test("invalid frame and reversed timeline rejected", () => {
  const data = {
    version: 1,
    frames: [
      { elapsed_ms: 20, source: "live", frame: frame() },
      { elapsed_ms: 10, source: "live", frame: frame() },
    ],
  };
  assert.throws(() => validateRecording(data, () => true));
  assert.throws(() =>
    validateRecording(
      { version: 1, frames: [{ elapsed_ms: 0, source: "live", frame: {} }] },
      () => false,
    ),
  );
});
test("bounded history and recording sizes are intentional", () => {
  assert.equal(MAX_HISTORY, 300);
  assert.equal(MAX_RECORDING_FRAMES, 600);
  assert.equal(MAX_RECORDING_BYTES, 30 * 1024 * 1024);
  assert.throws(() =>
    validateRecording(
      {
        version: 1,
        frames: Array(601).fill({
          elapsed_ms: 0,
          source: "live",
          frame: frame(),
        }),
      },
      () => true,
    ),
  );
});

test("summarizeFrame derives accuracy correctly with dynamic traversability fallback", () => {
  // Case 1: tracking_accuracy explicitly reported
  const f1 = frame();
  f1.system_stats.tracking_accuracy = 95.4;
  const s1 = summarizeFrame(f1);
  assert.equal(s1.accuracy, 95.4);
  assert.equal(s1.stats.accuracy, 95.4);

  // Case 2: accuracy key reported
  const f2 = frame();
  delete f2.system_stats.tracking_accuracy;
  f2.system_stats.accuracy = 82.1;
  const s2 = summarizeFrame(f2);
  assert.equal(s2.accuracy, 82.1);

  // Case 3: Both missing, fallback derived from cell traversability ratio (< 50)
  // frame() cells costs: [0, 50, 51, 180, 181, 255] -> cost < 50 is [0] (1 of 6 cells = 16.7%)
  const f3 = frame();
  delete f3.system_stats.tracking_accuracy;
  delete f3.system_stats.accuracy;
  const s3 = summarizeFrame(f3);
  assert.equal(s3.accuracy, 16.7);
  assert.equal(s3.stats.accuracy, 16.7);

  // Case 4: Missing stats and empty cells -> null
  const f4 = { cells: [], dynamic_objects: [] };
  const s4 = summarizeFrame(f4);
  assert.equal(s4.accuracy, null);
  assert.equal(s4.stats.accuracy, null);
});

