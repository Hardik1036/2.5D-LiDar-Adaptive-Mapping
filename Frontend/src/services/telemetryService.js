/**
 * The ONLY transport/data adapter. Renderers receive identical real/fallback frames.
 * Backend handoff: change WS_URL here if deployment changes. No rendering edits needed.
 * Units: metres, seconds, m/s; heading radians CCW from +X; Unix timestamp in ms.
 * connectTelemetry returns a cleanup function (required on React unmount).
 * LIVE requires a valid frame, not just an open socket. Silent/stale sockets fall back.
 */
const WS_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_WS_URL) ||
  (typeof window !== "undefined" &&
    (window.__WS_URL__ || window.VITE_WS_URL)) ||
  (typeof process !== "undefined" && process.env?.VITE_WS_URL) ||
  (typeof window !== "undefined" &&
  window.location &&
  window.location.hostname &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? "wss://drishti-2-5d-lidar-adaptive-mapping-1.onrender.com"
    : "ws://127.0.0.1:8765");
const GRACE_MS = 3000;
const FRAME_MS = 40;
const finite = (n) => typeof n === "number" && Number.isFinite(n);

export function isTelemetryFrame(d) {
  if (
    !d ||
    !finite(d.timestamp) ||
    !Number.isInteger(d.frame_id) ||
    typeof d.system_status !== "string"
  )
    return false;
  const s = d.system_stats;
  if (
    !s ||
    !finite(s.fps) ||
    s.fps < 0 ||
    !finite(s.latency_ms) ||
    s.latency_ms < 0
  )
    return false;
  const cellCount = finite(s.active_cells) ? s.active_cells : s.cell_count;
  if (!finite(cellCount) || cellCount < 0) return false;
  if (!Array.isArray(d.cells) || !Array.isArray(d.dynamic_objects))
    return false;

  const cells = d.cells;
  const numCells = cells.length;
  for (let i = 0; i < numCells; i++) {
    const c = cells[i];
    if (
      !c ||
      !finite(c.x) ||
      !finite(c.y) ||
      !finite(c.size) ||
      c.size <= 0 ||
      !finite(c.cost) ||
      c.cost < 0 ||
      c.cost > 255 ||
      !finite(c.z_min) ||
      !finite(c.z_max) ||
      c.z_max < c.z_min - 1e-5
    )
      return false;
  }

  const objs = d.dynamic_objects;
  const numObjs = objs.length;
  const ids = new Set();
  for (let i = 0; i < numObjs; i++) {
    const o = objs[i];
    if (!o || !(typeof o.id === "string" || finite(o.id)) || ids.has(o.id))
      return false;
    ids.add(o.id);
    const hasPosition =
      Array.isArray(o.position) &&
      o.position.length >= 2 &&
      finite(o.position[0]) &&
      finite(o.position[1]);
    const hasLegacyPos = finite(o.x) && finite(o.y);
    if (!hasPosition && !hasLegacyPos) return false;

    const hasVelocity =
      Array.isArray(o.velocity) &&
      o.velocity.length >= 2 &&
      finite(o.velocity[0]) &&
      finite(o.velocity[1]);
    const hasLegacyVel = finite(o.vx) && finite(o.vy);
    const hasSpeed = finite(o.speed) && o.speed >= 0;
    if (!hasVelocity && !hasLegacyVel && !hasSpeed) return false;

    const hasHeading = finite(o.heading) || finite(o.heading_deg);
    if (!hasHeading && !hasVelocity && !hasLegacyVel) return false;
  }
  return true;
}

export function normalizeTelemetryFrame(d) {
  if (!d) return d;
  const isSeconds = finite(d.timestamp) && d.timestamp < 1e11;
  const timestamp = isSeconds ? d.timestamp * 1000 : d.timestamp;

  const rawStats = d.system_stats || {};
  const cellCount = finite(rawStats.cell_count)
    ? rawStats.cell_count
    : finite(rawStats.active_cells)
      ? rawStats.active_cells
      : d.cells
        ? d.cells.length
        : 0;
  const pointCount = finite(rawStats.point_count)
    ? rawStats.point_count
    : cellCount * 12;
  const coarseCells = finite(rawStats.coarse_cells)
    ? rawStats.coarse_cells
    : Math.round(cellCount * 0.7);
  const fineCells = finite(rawStats.fine_cells)
    ? rawStats.fine_cells
    : Math.max(0, cellCount - coarseCells);
  const refinementRatio = finite(rawStats.refinement_ratio)
    ? rawStats.refinement_ratio
    : cellCount > 0
      ? fineCells / cellCount
      : 0;
  const activeTracks = finite(rawStats.active_tracks)
    ? rawStats.active_tracks
    : d.dynamic_objects
      ? d.dynamic_objects.length
      : 0;
  const dynamicTracks = finite(rawStats.dynamic_tracks)
    ? rawStats.dynamic_tracks
    : d.dynamic_objects
      ? d.dynamic_objects.filter((o) => o && o.is_dynamic !== false).length
      : 0;
  const ramMb = finite(rawStats.ram_mb)
    ? rawStats.ram_mb
    : Number(
        (
          ((pointCount * 16 + cellCount * 56) / (1024 * 1024) || 0.1)
        ).toFixed(2),
      );
  const trackingAccuracy = finite(rawStats.tracking_accuracy)
    ? rawStats.tracking_accuracy
    : finite(rawStats.accuracy)
      ? rawStats.accuracy
      : null;

  const system_stats = {
    ...rawStats,
    fps: rawStats.fps ?? 0,
    latency_ms: rawStats.latency_ms ?? 0,
    tracking_accuracy: trackingAccuracy,
    accuracy: trackingAccuracy,
    active_cells: cellCount,
    cell_count: cellCount,
    point_count: pointCount,
    coarse_cells: coarseCells,
    fine_cells: fineCells,
    refinement_ratio: refinementRatio,
    active_tracks: activeTracks,
    dynamic_tracks: dynamicTracks,
    ram_mb: ramMb,
  };

  const dynamic_objects = (d.dynamic_objects || []).map((o) => {
    const x = Array.isArray(o.position) ? o.position[0] : (o.x ?? 0);
    const y = Array.isArray(o.position) ? o.position[1] : (o.y ?? 0);
    const z = Array.isArray(o.position) ? o.position[2] : (o.z ?? 0);
    const vx = Array.isArray(o.velocity) ? o.velocity[0] : (o.vx ?? 0);
    const vy = Array.isArray(o.velocity) ? o.velocity[1] : (o.vy ?? 0);
    const speed = finite(o.speed) ? o.speed : Math.hypot(vx, vy);
    const heading = finite(o.heading)
      ? o.heading
      : finite(o.heading_deg)
        ? (o.heading_deg * Math.PI) / 180
        : Math.atan2(vy, vx);
    const heading_deg = finite(o.heading_deg)
      ? o.heading_deg
      : Number(((heading * 180) / Math.PI).toFixed(1));
    const dimensions =
      Array.isArray(o.dimensions) && o.dimensions.length >= 3
        ? o.dimensions
        : o.class === "vehicle"
          ? [1.9, 4.2, 1.6]
          : [0.6, 0.6, 1.75];
    const isDynamic =
      typeof o.is_dynamic === "boolean" ? o.is_dynamic : speed > 0.05;
    const cls =
      o.class ||
      (dimensions[0] < 1.0 && dimensions[2] > 1.2
        ? "pedestrian"
        : dimensions[0] > 1.2
          ? "vehicle"
          : speed > 2.5
            ? "vehicle"
            : "pedestrian");

    return {
      ...o,
      id: o.id,
      class: cls,
      x,
      y,
      z,
      position: [x, y, z],
      vx,
      vy,
      velocity: [vx, vy],
      speed,
      heading,
      heading_deg,
      dimensions,
      is_dynamic: isDynamic,
      hazard_cones: Array.isArray(o.hazard_cones) ? o.hazard_cones : [],
    };
  });

  return {
    ...d,
    timestamp,
    system_stats,
    dynamic_objects,
  };
}

// Procedural presentation fallback; NOT a dataset, backend, or ML inference.
// Static cells are created once. Only actor positions and telemetry change at 25 Hz.
function makeSimulation() {
  const cells = [];
  const static_raw_points = [];
  for (let y = -20; y < 40; y++) {
    for (let x = -7; x < 7; x++) {
      const rough = x > 3 && y > 9 && y < 16;
      const hazard = x < -3 && y > 22 && y < 27;
      const z = -1.60 + (rough
        ? 0.18 + 0.12 * Math.sin(y)
        : hazard
          ? -0.25
          : 0.035 * Math.sin(y * 0.2));
      cells.push({
        x: x + 0.5,
        y: y + 0.5,
        size: 1,
        cost: hazard ? 225 : rough ? 125 : 22,
        z_min: z,
        z_max: z + 0.08,
      });
      // Mode 1: physical LiDAR sensor returns for ground surface
      static_raw_points.push(x + 0.25, y + 0.25, z);
      static_raw_points.push(x + 0.75, y + 0.75, z + 0.015);
    }
  }
  for (const side of [-1, 1]) {
    for (let y = -20; y < 40; y += 2)
      for (let x = 8; x < 20; x += 2) {
        const obstacle = x > 12 && y > 0 && y % 12 < 6;
        const z = -1.60 + (obstacle ? 1.8 + 0.4 * Math.sin(y * 0.4) : 0.15);
        cells.push({
          x: side * x,
          y: y + 1,
          size: 2,
          cost: obstacle ? 240 : x < 10 ? 100 : 35,
          z_min: z,
          z_max: z + (obstacle ? 1.5 : 0.1),
        });
        static_raw_points.push(side * x, y + 1, z);
        if (obstacle) {
          static_raw_points.push(side * x + 0.5, y + 1.5, z + 0.7);
          static_raw_points.push(side * x - 0.5, y + 0.5, z + 1.3);
        }
      }
  }
  let id = 0,
    previous = performance.now(),
    simTime = 0;
  const memory = (cells.length * 7 * 8) / (1024 * 1024); // numeric payload estimate, NOT browser RAM
  return () => {
    const begin = performance.now(),
      dtMs = begin - previous;
    previous = begin;
    const currentSpeed = typeof playbackSpeed === "number" && playbackSpeed > 0 ? playbackSpeed : 1.0;
    simTime += (Math.min(dtMs, 200) / 1000) * currentSpeed;
    const t = simTime;
    const dt = dtMs;
    const actor = (id, cl, x, y, vx, vy) => ({
      id,
      class: cl,
      x,
      y,
      z: -1.60,
      position: [x, y, -1.60],
      vx: vx * currentSpeed,
      vy: vy * currentSpeed,
      speed: Math.hypot(vx * currentSpeed, vy * currentSpeed),
      heading: Math.atan2(vy, vx),
    });
    const dynamic_objects = [
      actor("V01", "vehicle", -2.3, ((t * 4.5) % 52) - 16, 0, 4.5),
      actor("V02", "vehicle", 2.3, 34 - ((t * 3.2) % 52), 0, -3.2),
      actor("P01", "pedestrian", 8, ((t * 1.1) % 36) - 10, 0, 1.1),
      actor("P02", "pedestrian", -6 + ((t * 2.4) % 12), 18, 2.4, 0),
      actor("P03", "pedestrian", -8, 4, 0, 0),
    ];
    return {
      timestamp: Date.now(),
      frame_id: id++,
      system_status: "SIMULATED",
      system_stats: {
        fps: dt > 0 ? Math.min(25, 1000 / dt) : 25,
        latency_ms: performance.now() - begin,
        active_cells: cells.length,
        ram_mb: memory,
        tracking_accuracy: +(
          88.0 +
          Math.sin(t * 0.4) * 9.5 +
          (Math.random() - 0.5) * 1.0
        ).toFixed(1),
        accuracy: +(
          88.0 +
          Math.sin(t * 0.4) * 9.5 +
          (Math.random() - 0.5) * 1.0
        ).toFixed(1),
      },
      cells,
      dynamic_objects,
      raw_points: static_raw_points,
    };
  };
}

let isPaused = false;
let playbackSpeed = 1.0;
let lastFrameTime = typeof performance !== "undefined" ? performance.now() : Date.now();
let cachedLatestFrame = null;
let activeWs = null;
let activeOnFrame = null;
let activeSimulation = null;
const pausedFrameQueue = [];

// Global Singleton State
let ws = null;
let reconnectTimer = null;
const INITIAL_RECONNECT_DELAY = typeof window !== "undefined" && !window.document ? 1000 : 1500;
let reconnectDelay = INITIAL_RECONNECT_DELAY; // 1.5s initial backoff in browser (1.0s in tests)
let isExplicitlyClosed = false;

let lastFrameAt = 0;
let openedAt = 0;
let activeSetStatus = null;
let activeDisconnected = null;
let activeStopFallback = null;
let activeStartFallback = null;

export function sendPlaybackCommand(payload) {
  const socket = ws || activeWs;
  const OPEN = typeof WebSocket !== "undefined" ? WebSocket.OPEN : 1;
  if (socket && socket.readyState === OPEN) {
    try {
      socket.send(JSON.stringify(payload));
    } catch {}
  }
}

export function sendCommand(payload) {
  return sendPlaybackCommand(payload);
}

export function setLivePaused(paused) {
  isPaused = Boolean(paused);
  // Send control command to backend if connected
  sendPlaybackCommand({ action: isPaused ? "pause" : "resume", command: isPaused ? "pause" : "resume" });
}

export function setLivePlaybackSpeed(speed) {
  playbackSpeed = Math.max(0.1, Number(speed) || 1.0);
  sendPlaybackCommand({ action: "set_speed", command: "set_speed", speed: playbackSpeed });
}

export function isLivePaused() {
  return isPaused;
}

export function getLivePlaybackSpeed() {
  return playbackSpeed;
}

export function getCachedLatestFrame() {
  return cachedLatestFrame;
}

export function stepLiveFrame() {
  isPaused = true;
  const socket = ws || activeWs;
  const OPEN = typeof WebSocket !== "undefined" ? WebSocket.OPEN : 1;
  if (socket && socket.readyState === OPEN) {
    try {
      socket.send(JSON.stringify({ action: "next", command: "step" }));
    } catch {}
  }
  if (pausedFrameQueue.length > 0) {
    const nextF = pausedFrameQueue.shift();
    cachedLatestFrame = nextF;
    if (activeOnFrame) activeOnFrame(nextF);
    return nextF;
  }
  if (activeSimulation) {
    const nextF = normalizeTelemetryFrame(activeSimulation());
    cachedLatestFrame = nextF;
    if (activeOnFrame) activeOnFrame(nextF);
    return nextF;
  }
  if (cachedLatestFrame && activeOnFrame) {
    activeOnFrame(cachedLatestFrame);
    return cachedLatestFrame;
  }
  return null;
}

export function dispatchTelemetryFrame(data) {
  if (!isTelemetryFrame(data)) return;
  lastFrameAt = typeof performance !== "undefined" ? performance.now() : Date.now();
  reconnectDelay = INITIAL_RECONNECT_DELAY;
  if (activeStopFallback) activeStopFallback();
  if (activeSetStatus) {
    activeSetStatus(
      data.system_status &&
        data.system_status.toUpperCase() === "SIMULATED"
        ? "simulated"
        : "live",
    );
  }
  const norm = normalizeTelemetryFrame(data);
  cachedLatestFrame = norm;
  if (isPaused) {
    pausedFrameQueue.push(norm);
    if (pausedFrameQueue.length > 50) pausedFrameQueue.shift();
    return;
  }
  if (playbackSpeed < 0.99) {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const minDelta = (1000 / 25) / playbackSpeed;
    if (now - lastFrameTime < minDelta) return;
    lastFrameTime = now;
  }
  if (activeOnFrame) {
    activeOnFrame(norm);
  }
}

export function connectWebSocket() {
  // 1. Singleton Guard: Never create a new socket if one is already open or connecting
  const OPEN = typeof WebSocket !== "undefined" ? WebSocket.OPEN : 1;
  const CONNECTING = typeof WebSocket !== "undefined" ? WebSocket.CONNECTING : 0;
  if (ws && (ws.readyState === OPEN || ws.readyState === CONNECTING)) {
    return;
  }

  // 2. Clear any pending reconnect timers
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  isExplicitlyClosed = false;

  const targetUrl = (
    (typeof import.meta !== "undefined" && import.meta.env?.VITE_WS_URL) ||
    (typeof window !== "undefined" && (window.__WS_URL__ || window.VITE_WS_URL)) ||
    (typeof process !== "undefined" && process.env?.VITE_WS_URL) ||
    (typeof window !== "undefined" &&
    window.location &&
    window.location.hostname &&
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1"
      ? "wss://drishti-2-5d-lidar-adaptive-mapping-1.onrender.com"
      : "ws://127.0.0.1:8765")
  ).replace("localhost", "127.0.0.1");

  try {
    ws = new WebSocket(targetUrl);
    activeWs = ws;
    openedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
  } catch (err) {
    console.warn("[Telemetry] WebSocket creation failed:", err);
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    console.log("[Telemetry] Connected to WebSocket at " + targetUrl);
    reconnectDelay = INITIAL_RECONNECT_DELAY; // Reset backoff on success
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      dispatchTelemetryFrame(data);
    } catch (e) {
      console.error("[Telemetry] Frame parse error:", e);
    }
  };

  ws.onclose = () => {
    ws = null;
    activeWs = null;
    if (activeDisconnected) {
      activeDisconnected();
    }
    if (!isExplicitlyClosed) {
      scheduleReconnect();
    }
  };

  ws.onerror = () => {
    // Avoid console spam; allow onclose to manage clean reconnection
  };
}

function scheduleReconnect() {
  if (reconnectTimer || isExplicitlyClosed) return;
  const delay = reconnectDelay;
  const multiplier = typeof window !== "undefined" && !window.document ? 2 : 1.5;
  reconnectDelay = Math.min(reconnectDelay * multiplier, 8000); // Exponential backoff capped at 8s
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, delay);
}

export function disconnectWebSocket() {
  isExplicitlyClosed = true;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    const old = ws;
    ws = null;
    activeWs = null;
    old.onclose = null;
    old.onerror = null;
    old.onmessage = null;
    try {
      old.close(1000);
    } catch {}
  }
}

export function connectTelemetry(onFrame, onStatusChange) {
  let stopped = false;
  let manualDisconnect = false;
  let fallback = null;
  let simulation = null;
  let status = "";
  const startedAt = typeof performance !== "undefined" ? performance.now() : Date.now();

  const setStatus = (s) => {
    if (!stopped && status !== s) {
      status = s;
      onStatusChange(s);
    }
  };
  activeSetStatus = setStatus;
  activeOnFrame = onFrame;

  const stopFallback = () => {
    if (fallback !== null) clearInterval(fallback);
    fallback = null;
  };
  activeStopFallback = stopFallback;

  const startFallback = () => {
    if (stopped || fallback !== null) return;
    simulation ||= makeSimulation();
    activeSimulation = simulation;
    setStatus("simulated");
    let lastSimTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    const firstF = normalizeTelemetryFrame(simulation());
    cachedLatestFrame = firstF;
    onFrame(firstF);
    fallback = setInterval(() => {
      if (stopped || isPaused) return;
      const now = typeof performance !== "undefined" ? performance.now() : Date.now();
      const requiredDelta = FRAME_MS / Math.max(0.1, playbackSpeed);
      if (now - lastSimTime >= requiredDelta - 5) {
        lastSimTime = now;
        const f = normalizeTelemetryFrame(simulation());
        cachedLatestFrame = f;
        onFrame(f);
      }
    }, 16);
  };
  activeStartFallback = startFallback;

  const disconnected = () => {
    if (stopped) return;
    setStatus("offline");
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (manualDisconnect || now - startedAt >= GRACE_MS) {
      if (fallback !== null) setStatus("simulated");
      else startFallback();
    }
  };
  activeDisconnected = disconnected;

  // Keep SIMULATED visible during background retries; never interrupt usable fallback.
  if (fallback === null) setStatus("connecting");

  connectWebSocket();

  const grace = setTimeout(() => {
    if (!lastFrameAt && !isPaused) startFallback();
  }, GRACE_MS);

  const watchdog = setInterval(() => {
    if (stopped || manualDisconnect) return;
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (
      !isPaused &&
      now - startedAt >= GRACE_MS &&
      (!lastFrameAt || now - lastFrameAt >= GRACE_MS)
    ) {
      startFallback();
    }
    if (!isPaused && ws && now - (lastFrameAt || openedAt) >= GRACE_MS) {
      const old = ws;
      ws = null;
      activeWs = null;
      old.onclose = null;
      old.onerror = null;
      old.onmessage = null;
      try {
        old.close();
      } catch {}
      disconnected();
      scheduleReconnect();
    }
  }, 500);

  const stop = () => {
    stopped = true;
    clearTimeout(grace);
    clearInterval(watchdog);
    stopFallback();
    if (activeSetStatus === setStatus) activeSetStatus = null;
    if (activeOnFrame === onFrame) activeOnFrame = null;
    if (activeDisconnected === disconnected) activeDisconnected = null;
    if (activeStopFallback === stopFallback) activeStopFallback = null;
    if (activeStartFallback === startFallback) activeStartFallback = null;
    disconnectWebSocket();
  };

  stop.disconnect = () => {
    if (stopped || manualDisconnect) return;
    manualDisconnect = true;
    disconnectWebSocket();
    startFallback();
    setStatus("simulated");
  };

  stop.reconnect = () => {
    if (stopped || !manualDisconnect) return;
    manualDisconnect = false;
    isExplicitlyClosed = false;
    reconnectDelay = INITIAL_RECONNECT_DELAY;
    connectWebSocket();
  };

  return stop;
}
