import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const source = fs
  .readFileSync(
    new URL("../src/services/telemetryService.js", import.meta.url),
    "utf8",
  )
  .replaceAll("export function", "function")
  .replaceAll("import.meta", "({ env: {} })");
function harness() {
  let now = 0,
    next = 0;
  const timers = new Map(),
    sockets = [],
    frames = [],
    statuses = [];
  const add = (fn, delay, period = 0) => {
    const id = ++next;
    timers.set(id, { fn, at: now + delay, period });
    return id;
  };
  class Socket {
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      sockets.push(this);
    }
    close() {
      this.readyState = 3;
      this.onclose?.({});
    }
    send(data) {
      this.readyState = 1;
      this.onmessage?.({
        data: typeof data === "string" ? data : JSON.stringify(data),
      });
    }
  }
  const context = {
    window: { location: { hostname: "localhost" } },
    WebSocket: Socket,
    performance: { now: () => now },
    Date,
    console,
    setTimeout: (f, d) => add(f, d),
    setInterval: (f, d) => add(f, d, d),
    clearTimeout: (id) => timers.delete(id),
    clearInterval: (id) => timers.delete(id),
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  function tick(ms) {
    const end = now + ms;
    let guard = 0;
    while (true) {
      if (guard++ > 100000) throw Error("Timer loop");
      let nextId = null,
        event = null;
      for (const [id, t] of timers)
        if (t.at <= end && (!event || t.at < event.at)) {
          nextId = id;
          event = t;
        }
      if (!event) break;
      now = event.at;
      if (event.period) event.at += event.period;
      else timers.delete(nextId);
      event.fn();
    }
    now = end;
  }
  const stop = context.connectTelemetry(
    (f) => frames.push(f),
    (s) => statuses.push(s),
  );
  return {
    tick,
    stop,
    sockets,
    frames,
    statuses,
    timers,
    validate: context.isTelemetryFrame,
    normalize: context.normalizeTelemetryFrame,
  };
}
const valid = () => ({
  timestamp: Date.now(),
  frame_id: 1,
  system_status: "LIVE",
  system_stats: { fps: 25, latency_ms: 8, active_cells: 1, ram_mb: 0.1 },
  cells: [{ x: 0, y: 0, size: 1, cost: 50, z_min: 0, z_max: 1 }],
  dynamic_objects: [
    { id: 1, class: "person", x: 0, y: 0, vx: 0, vy: 0, speed: 0, heading: 0 },
  ],
});
test("starts connecting at correct URL; fallback starts at 3 seconds, not earlier", () => {
  const h = harness();
  assert.equal(h.statuses[0], "connecting");
  assert.equal(h.sockets[0].url, "ws://127.0.0.1:8765");
  h.tick(2999);
  assert.equal(h.frames.length, 0);
  h.tick(1);
  assert.equal(h.statuses.at(-1), "simulated");
  assert.ok(h.frames.length);
  assert.ok(h.validate(h.frames[0]));
  assert.ok(h.frames[0].cells.length > 1000);
  const cells = h.frames[0].cells;
  h.tick(80);
  assert.equal(h.frames.at(-1).cells, cells);
  h.stop();
});
test("valid live data promotes fallback without synthetic interleaving", () => {
  const h = harness();
  h.tick(4100);
  h.sockets.at(-1).send(valid());
  assert.equal(h.statuses.at(-1), "live");
  const n = h.frames.length;
  h.tick(200);
  assert.equal(h.frames.length, n);
  h.stop();
});
test("malformed JSON, NaN, invalid cells and duplicate IDs rejected", () => {
  const h = harness();
  h.sockets[0].send("{bad");
  h.sockets[0].send({});
  let f = valid();
  f.cells[0].size = -1;
  h.sockets[0].send(f);
  f = valid();
  f.dynamic_objects.push({ ...f.dynamic_objects[0] });
  h.sockets[0].send(f);
  assert.equal(h.frames.length, 0);
  f = valid();
  f.system_stats.fps = NaN;
  assert.equal(h.validate(f), false);
  h.stop();
});
test("empty arrays and zero metrics are legitimate live frames", () => {
  const h = harness(),
    f = valid();
  f.cells = [];
  f.dynamic_objects = [];
  f.system_stats = { fps: 0, latency_ms: 0, active_cells: 0, ram_mb: 0 };
  h.sockets[0].send(f);
  assert.equal(h.statuses.at(-1), "live");
  assert.equal(h.frames[0].system_stats.ram_mb, 0);
  h.stop();
});
test("silent open socket falls back and reconnects", () => {
  const h = harness();
  h.sockets[0].readyState = 1;
  h.tick(4100);
  assert.equal(h.statuses.at(-1), "simulated");
  assert.ok(h.sockets.length >= 2);
  h.stop();
});
test("live disconnect immediately resumes simulation after startup grace", () => {
  const h = harness();
  h.tick(100);
  h.sockets[0].send(valid());
  h.tick(1000);
  h.sockets[0].send(valid());
  h.tick(2000);
  h.sockets[0].close();
  assert.ok(h.statuses.includes("offline"));
  assert.equal(h.statuses.at(-1), "simulated");
  h.stop();
});
test("stale live frame does not remain LIVE indefinitely", () => {
  const h = harness();
  h.tick(10);
  h.sockets[0].send(valid());
  h.tick(3490);
  assert.equal(h.statuses.at(-1), "simulated");
  h.stop();
});
test("exponential reconnection waits 1s, 2s, 4s", () => {
  const h = harness();
  h.sockets.at(-1).close();
  h.tick(999);
  assert.equal(h.sockets.length, 1);
  h.tick(1);
  assert.equal(h.sockets.length, 2);
  h.sockets.at(-1).close();
  h.tick(1999);
  assert.equal(h.sockets.length, 2);
  h.tick(1);
  assert.equal(h.sockets.length, 3);
  h.sockets.at(-1).close();
  h.tick(3999);
  assert.equal(h.sockets.length, 3);
  h.tick(1);
  assert.equal(h.sockets.length, 4);
  h.stop();
});
test("cleanup cancels every timer and ignores late messages", () => {
  const h = harness();
  h.tick(4500);
  const socket = h.sockets.at(-1);
  h.stop();
  assert.equal(h.timers.size, 0);
  const n = h.frames.length;
  socket.send(valid());
  h.tick(20000);
  assert.equal(h.frames.length, n);
  assert.equal(h.timers.size, 0);
});

const pythonBackendFrame = () => ({
  timestamp: 1725612345.123,
  frame_id: 142,
  system_status: "ALL_SYSTEMS_NOMINAL",
  system_stats: {
    fps: 28.4,
    latency_ms: 18.2,
    point_count: 21850,
    cell_count: 1840,
    coarse_cells: 1280,
    fine_cells: 560,
    refinement_ratio: 0.304,
    active_tracks: 2,
    dynamic_tracks: 2,
  },
  cells: [
    {
      x: 4.25,
      y: -2.75,
      size: 0.5,
      cost: 15,
      z_min: -1.62,
      z_max: -1.58,
      delta_z: 0.04,
      variance: 0.0008,
      slope: 3.2,
      pts: 28,
    },
  ],
  dynamic_objects: [
    {
      id: 1,
      position: [4.0, -6.4, -0.72],
      velocity: [0.0, 1.2],
      speed: 1.2,
      heading: 1.57,
      heading_deg: 90.0,
      dimensions: [0.5, 0.5, 1.75],
      is_dynamic: true,
      hazard_cones: [
        {
          t: 1.0,
          center: [4.0, -5.2],
          semi_major: 0.58,
          semi_minor: 0.45,
          heading_deg: 90.0,
          polygon: [
            [3.8, -5.5],
            [4.2, -5.5],
            [4.1, -4.9],
            [3.9, -4.9],
          ],
        },
      ],
    },
  ],
});

test("accepts and validates Python LiDAR perception backend frame payload", () => {
  const h = harness();
  const raw = pythonBackendFrame();
  assert.equal(h.validate(raw), true);

  h.sockets[0].send(raw);
  assert.equal(h.statuses.at(-1), "live");
  assert.equal(h.frames.length, 1);

  const f = h.frames[0];
  // Timestamp normalized from seconds to ms
  assert.equal(f.timestamp, 1725612345123);
  assert.equal(f.frame_id, 142);
  assert.equal(f.system_status, "ALL_SYSTEMS_NOMINAL");

  // Quadtree and perception stats preserved
  assert.equal(f.system_stats.fps, 28.4);
  assert.equal(f.system_stats.latency_ms, 18.2);
  assert.equal(f.system_stats.cell_count, 1840);
  assert.equal(f.system_stats.active_cells, 1840);
  assert.equal(f.system_stats.point_count, 21850);
  assert.equal(f.system_stats.coarse_cells, 1280);
  assert.equal(f.system_stats.fine_cells, 560);
  assert.equal(f.system_stats.refinement_ratio, 0.304);
  assert.equal(f.system_stats.active_tracks, 2);
  assert.equal(f.system_stats.dynamic_tracks, 2);
  assert.ok(f.system_stats.ram_mb > 0);

  // Dynamic object mapping and unpack
  assert.equal(f.dynamic_objects.length, 1);
  const obj = f.dynamic_objects[0];
  assert.equal(obj.id, 1);
  assert.equal(obj.x, 4.0);
  assert.equal(obj.y, -6.4);
  assert.equal(obj.z, -0.72);
  assert.equal(obj.vx, 0.0);
  assert.equal(obj.vy, 1.2);
  assert.equal(obj.speed, 1.2);
  assert.equal(obj.heading, 1.57);
  assert.equal(obj.heading_deg, 90.0);
  assert.equal(obj.is_dynamic, true);
  assert.equal(obj.class, "pedestrian"); // Inferred from dimensions [0.5, 0.5, 1.75]
  assert.equal(obj.hazard_cones.length, 1);
  assert.equal(obj.hazard_cones[0].t, 1.0);
  assert.equal(obj.hazard_cones[0].center[0], 4.0);
  assert.equal(obj.hazard_cones[0].center[1], -5.2);
  h.stop();
});

test("correctly converts timestamps and vehicle dimensions in normalization", () => {
  const h = harness();
  const raw = pythonBackendFrame();
  raw.dynamic_objects[0].dimensions = [1.8, 4.5, 1.5];
  delete raw.dynamic_objects[0].heading;
  raw.dynamic_objects[0].heading_deg = 180.0;
  const normalized = h.normalize(raw);
  assert.equal(normalized.dynamic_objects[0].class, "vehicle");
  assert.ok(Math.abs(normalized.dynamic_objects[0].heading - Math.PI) < 0.01);
  h.stop();
});

test("supports custom endpoint via environment / window override", () => {
  let now = 0;
  const timers = new Map();
  const sockets = [];
  class CustomSocket {
    constructor(url) {
      this.url = url;
      sockets.push(this);
    }
    close() {}
  }
  const customContext = {
    window: { __WS_URL__: "ws://192.168.1.100:9999", location: { hostname: "example.com" } },
    WebSocket: CustomSocket,
    performance: { now: () => now },
    Date,
    console,
    setTimeout: (f, d) => timers.set(1, { f }),
    setInterval: () => 1,
    clearTimeout: () => {},
    clearInterval: () => {},
  };
  vm.createContext(customContext);
  vm.runInContext(source, customContext);
  const stop = customContext.connectTelemetry(() => {}, () => {});
  assert.equal(sockets[0].url, "ws://192.168.1.100:9999");
  stop();
});

test("manual disconnect closes active socket, switches to simulation, and reconnects", () => {
  const h = harness();
  h.tick(100);
  assert.equal(h.statuses[0], "connecting");
  assert.ok(h.sockets.length > 0);

  // Call manual disconnect
  h.stop.disconnect();
  assert.equal(h.statuses.at(-1), "simulated");
  assert.ok(h.frames.length > 0);

  // Ensure no automatic background reconnection happens while manually disconnected
  const socketCount = h.sockets.length;
  h.tick(10000);
  assert.equal(h.sockets.length, socketCount);

  // Call manual reconnect
  h.stop.reconnect();
  assert.ok(h.sockets.length > socketCount);
  h.stop();
});


