import React, { useEffect, useState } from "react";
import Header from "./components/Header.jsx";
import SceneControls from "./components/SceneControls.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import SceneViewport from "./components/SceneViewport.jsx";
import Settings from "./components/Settings.jsx";
import { Icon } from "./components/Brand.jsx";
import Welcome from "./pages/Welcome.jsx";
import Analysis from "./pages/Analysis.jsx";
import Playback from "./pages/Playback.jsx";
import SystemLoader from "./components/SystemLoader.jsx";
import { useTelemetry } from "./hooks/useTelemetry.js";
import { DEFAULT_OPTIONS } from "./three/Scene.js";
const route = () =>
  ["live", "analysis", "playback"].includes(location.hash.slice(1))
    ? location.hash.slice(1)
    : "welcome";
export default function App() {
  const [initialized, setInitialized] = useState(false);
  const [page, setPage] = useState(route),
    [settings, setSettings] = useState(false),
    [options, setOptions] = useState({
      ...DEFAULT_OPTIONS,
      pixelRatio: 1.5,
      motion: true,
    });
  const telemetry = useTelemetry();
  useEffect(() => {
    const changed = () => {
      setPage(route());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  const navigate = (p) => {
    location.hash = p;
    setPage(p);
    window.scrollTo(0, 0);
  };
  const start = () => {
    if (
      telemetry.session.length &&
      !window.confirm(
        "Start a new recording? This replaces the previous recording. Export it first if needed.",
      )
    )
      return;
    telemetry.startRecording();
  };
  useEffect(() => {
    const handleKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.code === "Space") {
        e.preventDefault();
        telemetry.togglePause();
      } else if (e.key === "n" || e.key === "N" || e.key === "ArrowRight") {
        if (e.key === "ArrowRight" && e.altKey) return;
        telemetry.stepFrame();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [telemetry.togglePause, telemetry.stepFrame]);

  return (
    <div className={`premium-app page-${page}`}>
      {!initialized && <SystemLoader onComplete={() => setInitialized(true)} />}
      <Header
        status={telemetry.status}
        connectionState={telemetry.connectionState}
        page={page}
        navigate={navigate}
        onSettings={() => setSettings(true)}
        isManuallyDisconnected={telemetry.isManuallyDisconnected}
        onToggleBackend={telemetry.toggleBackend}
      />
      {page === "welcome" ? (
        <Welcome onLaunch={() => navigate("live")} />
      ) : page === "live" ? (
        <>
          <div className="live-toolbar">
            <div>
              <span className="overline">LIVE WORKSPACE</span>
              <span>Adaptive terrain &amp; dynamic perception</span>
            </div>
            <div className="page-actions">
              <button
                className={`secondary-button ${telemetry.isPaused ? "paused-state-btn" : ""}`}
                onClick={telemetry.togglePause}
                title={telemetry.isPaused ? "Resume real-time stream (Space)" : "Pause stream (Space)"}
              >
                <Icon name={telemetry.isPaused ? "playback" : "pause"} size={16} />
                {telemetry.isPaused ? "Play" : "Pause"}
              </button>
              <button
                className="secondary-button"
                onClick={telemetry.stepFrame}
                title="Step to next sweep / frame (N or →)"
              >
                <Icon name="step" size={16} />
                Next
              </button>
              {telemetry.recording && (
                <span className="recording-state">
                  <i />
                  {telemetry.session.length} frames captured
                </span>
              )}
              <button
                className={`secondary-button ${telemetry.recording ? "record-active" : ""}`}
                disabled={!telemetry.summary}
                onClick={telemetry.recording ? telemetry.stopRecording : start}
              >
                <Icon name={telemetry.recording ? "stop" : "record"} />
                {telemetry.recording ? "Stop recording" : "Record session"}
              </button>
              <button
                className="text-button"
                onClick={() => navigate("playback")}
              >
                Review <Icon name="arrow" />
              </button>
            </div>
          </div>
          {telemetry.notice && (
            <div className="notice-banner" role="status">
              {telemetry.notice}
              <button
                className="icon-button"
                onClick={() => telemetry.setNotice("")}
                aria-label="Dismiss notice"
              >
                <Icon name="close" />
              </button>
            </div>
          )}
          <main className="workspace">
            <SceneControls
              options={options}
              setOptions={setOptions}
              isPaused={telemetry.isPaused}
              onTogglePause={telemetry.togglePause}
              onStepFrame={telemetry.stepFrame}
              playbackSpeed={telemetry.playbackSpeed}
              onSpeedChange={telemetry.changePlaybackSpeed}
              onReset={() =>
                window.dispatchEvent(new KeyboardEvent("keydown", { key: "r" }))
              }
            />
            <SceneViewport
              frameRef={telemetry.latest}
              summary={telemetry.summary}
              options={options}
              onModeChange={(mode) => setOptions((p) => ({ ...p, mode }))}
            />
            <MetricsPanel
              summary={telemetry.summary}
              status={telemetry.status}
              options={options}
            />
          </main>
        </>
      ) : page === "analysis" ? (
        <Analysis
          history={telemetry.history}
          summary={telemetry.summary}
          latest={telemetry.latest}
          latestFrame={telemetry.latest?.current}
          frame={telemetry.latest?.current}
          status={telemetry.status}
        />
      ) : (
        <Playback
          session={telemetry.session}
          replaceSession={telemetry.replaceSession}
          recording={telemetry.recording}
          options={options}
          navigate={navigate}
        />
      )}
      {page !== "welcome" && (
        <footer className="footer">
          <span>
            <i className={`footer-dot ${telemetry.status}`} />
            {page === "playback"
              ? "Recorded-view workspace"
              : telemetry.isManuallyDisconnected
                ? "Backend manually disconnected · Procedural simulation active (Pipeline paused)"
                : telemetry.status === "live" || telemetry.status === "connected"
                  ? "Telemetry Stream (CONNECTED · Drishti Core Pipeline)"
                  : telemetry.status === "simulated"
                    ? "Simulation fallback active · retrying backend connection"
                    : telemetry.status === "connecting"
                      ? "Connecting to Drishti perception pipeline..."
                      : "DISCONNECTED · retrying backend connection"}
          </span>
          <span>Coordinates: metres · Heading: radians · Z-up</span>
        </footer>
      )}
      {settings && (
        <Settings
          onClose={() => setSettings(false)}
          options={options}
          setOptions={setOptions}
        />
      )}
    </div>
  );
}
