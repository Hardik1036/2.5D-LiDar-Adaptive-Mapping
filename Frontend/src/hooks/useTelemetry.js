import { useEffect, useRef, useState } from "react";
import {
  connectTelemetry,
  isLivePaused,
  setLivePaused,
  getLivePlaybackSpeed,
  setLivePlaybackSpeed,
  stepLiveFrame,
} from "../services/telemetryService.js";
import {
  summarizeFrame,
  MAX_HISTORY,
  MAX_RECORDING_BYTES,
  MAX_RECORDING_FRAMES,
} from "../utils/session.js";
export { summarizeFrame };

const DEFAULT_WS_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_WS_URL) ||
  (typeof window !== "undefined" && (window.__WS_URL__ || window.VITE_WS_URL)) ||
  (typeof process !== "undefined" && process.env?.VITE_WS_URL) ||
  (typeof window !== "undefined" &&
  window.location &&
  window.location.hostname &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? "wss://drishti-2-5d-lidar-adaptive-mapping-1.onrender.com"
    : "ws://127.0.0.1:8765");

export function useTelemetry(customUrl) {
  const rawUrl = customUrl || (typeof window !== "undefined" && window.__WS_URL__) || DEFAULT_WS_URL;
  const wsUrl = rawUrl.replace("localhost", "127.0.0.1");
  const latest = useRef(null),
    historyRef = useRef([]),
    capture = useRef({
      active: false,
      frames: [],
      bytes: 0,
      start: 0,
      last: -Infinity,
    });
  const [status, setStatus] = useState("connecting"),
    [summary, setSummary] = useState(null),
    [history, setHistory] = useState([]),
    [session, setSession] = useState([]),
    [recording, setRecording] = useState(false),
    [notice, setNotice] = useState(""),
    [isPaused, setIsPaused] = useState(() => isLivePaused()),
    [playbackSpeed, setSpeed] = useState(() => getLivePlaybackSpeed());
  const telemetryControl = useRef(null);
  const userDisconnectedRef = useRef(false);
  const [isManuallyDisconnected, setIsManuallyDisconnected] = useState(false);
  const source = useRef("connecting");
  useEffect(() => {
    let lastHistory = -Infinity,
      lastShown = null,
      lastHudSync = 0,
      rafId = null,
      timerId = null,
      rafScheduled = false;

    const HUD_INTERVAL_MS = 100; // Throttled React dispatch ~10 Hz

    const syncHud = () => {
      rafScheduled = false;
      lastHudSync = performance.now();
      const f = latest.current;
      if (f && f !== lastShown) {
        lastShown = f;
        setSummary(summarizeFrame(f));
      }
      if (capture.current.active) {
        setSession([...capture.current.frames]);
      }
    };

    const scheduleHud = () => {
      const now = performance.now();
      if (rafScheduled) return;

      if (now - lastHudSync >= HUD_INTERVAL_MS) {
        rafScheduled = true;
        rafId = requestAnimationFrame(syncHud);
      } else if (!timerId) {
        timerId = setTimeout(() => {
          timerId = null;
          if (!rafScheduled) {
            rafScheduled = true;
            rafId = requestAnimationFrame(syncHud);
          }
        }, Math.max(0, HUD_INTERVAL_MS - (now - lastHudSync)));
      }
    };

    const stop = connectTelemetry(
      (f) => {
        latest.current = f;
        const now = performance.now();
        if (now - lastHistory >= 1000) {
          lastHistory = now;
          const list = historyRef.current;
          list.push({
            at: Date.now(),
            source: source.current,
            ...f.system_stats,
            objects: f.dynamic_objects.length,
          });
          if (list.length > MAX_HISTORY) list.shift();
          setHistory([...list]);
        }
        const c = capture.current;
        if (c.active && now - c.last >= 100) {
          const bytes =
            new TextEncoder().encode(JSON.stringify(f)).length + 100;
          if (
            c.frames.length >= MAX_RECORDING_FRAMES ||
            c.bytes + bytes > MAX_RECORDING_BYTES
          ) {
            c.active = false;
            setRecording(false);
            setNotice(
              "Recording stopped at the data/frame limit. Export it before starting another.",
            );
            setSession([...c.frames]);
            return;
          }
          c.frames.push({
            elapsed_ms: now - c.start,
            source: source.current === "live" ? "live" : "simulated",
            frame: f,
          });
          c.bytes += bytes;
          c.last = now;
        }
        scheduleHud();
      },
      (s) => {
        source.current = s;
        setStatus(s);
      },
    );
    telemetryControl.current = stop;

    return () => {
      telemetryControl.current = null;
      stop();
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (timerId !== null) clearTimeout(timerId);
    };
  }, []);
  function disconnectBackend() {
    userDisconnectedRef.current = true;
    setIsManuallyDisconnected(true);
    telemetryControl.current?.disconnect?.();
    setNotice("Backend manually disconnected. Procedural simulation fallback is active.");
  }
  function reconnectBackend() {
    userDisconnectedRef.current = false;
    setIsManuallyDisconnected(false);
    telemetryControl.current?.reconnect?.();
    setNotice("Reconnecting to LiDAR perception backend (ws://127.0.0.1:8765)...");
  }
  function toggleBackend() {
    if (userDisconnectedRef.current) {
      reconnectBackend();
    } else {
      disconnectBackend();
    }
  }
  function startRecording() {
    capture.current = {
      active: true,
      frames: [],
      bytes: 0,
      start: performance.now(),
      last: -Infinity,
    };
    setSession([]);
    setNotice("");
    setRecording(true);
  }
  function stopRecording() {
    capture.current.active = false;
    setRecording(false);
    setSession([...capture.current.frames]);
  }
  function replaceSession(frames) {
    capture.current.active = false;
    setRecording(false);
    capture.current.frames = frames;
    setSession(frames);
    setNotice("Recording imported locally. No file was uploaded to a server.");
  }
  function togglePause() {
    const next = !isLivePaused();
    setLivePaused(next);
    setIsPaused(next);
  }

  function pausePlayback() {
    setLivePaused(true);
    setIsPaused(true);
  }

  function resumePlayback() {
    setLivePaused(false);
    setIsPaused(false);
  }

  function stepFrame() {
    setLivePaused(true);
    setIsPaused(true);
    const nextF = stepLiveFrame();
    if (nextF) {
      latest.current = nextF;
      setSummary(summarizeFrame(nextF));
    }
  }

  function changePlaybackSpeed(val) {
    const s = Math.max(0.1, Math.min(10.0, Number(val) || 1.0));
    setLivePlaybackSpeed(s);
    setSpeed(s);
  }

  const connectionState =
    isManuallyDisconnected
      ? "SIMULATED (MANUAL)"
      : status === "live" || status === "connected"
        ? "CONNECTED"
        : status === "connecting"
          ? "CONNECTING"
          : status === "simulated"
            ? "SIMULATED"
            : "DISCONNECTED";

  return {
    latest,
    status,
    connectionState,
    summary,
    history,
    session,
    recording,
    notice,
    isManuallyDisconnected,
    isPaused,
    playbackSpeed,
    togglePause,
    pausePlayback,
    resumePlayback,
    stepFrame,
    changePlaybackSpeed,
    disconnectBackend,
    reconnectBackend,
    toggleBackend,
    startRecording,
    stopRecording,
    replaceSession,
    setNotice,
  };
}
