import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../components/Brand.jsx";
import SceneViewport from "../components/SceneViewport.jsx";
import {
  summarizeFrame,
  validateRecording,
  downloadText,
  MAX_RECORDING_BYTES,
} from "../utils/session.js";
import { isTelemetryFrame } from "../services/telemetryService.js";
export default function Playback({
  session,
  replaceSession,
  recording,
  options,
  navigate,
}) {
  const [index, setIndex] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [error, setError] = useState("");
  const input = useRef(null),
    identity = useRef(session[0]);
  useEffect(() => {
    if (identity.current !== session[0]) {
      identity.current = session[0];
      setIndex(0);
      setPlaying(false);
    }
  }, [session]);
  useEffect(() => {
    if (!playing || !session.length) return;
    if (index >= session.length - 1) {
      setPlaying(false);
      return;
    }
    const delay = Math.max(
      16,
      (session[index + 1].elapsed_ms - session[index].elapsed_ms) / speed,
    );
    const t = setTimeout(
      () => setIndex((i) => Math.min(i + 1, session.length - 1)),
      delay,
    );
    return () => clearTimeout(t);
  }, [playing, index, speed, session]);
  const entry = session[Math.min(index, Math.max(0, session.length - 1))],
    summary = entry ? summarizeFrame(entry.frame) : null;
  const duration = session.length
      ? (session.at(-1).elapsed_ms - session[0].elapsed_ms) / 1000
      : 0,
    elapsed = entry ? (entry.elapsed_ms - session[0].elapsed_ms) / 1000 : 0;
  async function importFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setError("");
    if (recording) {
      setError("Stop recording before importing another session.");
      return;
    }
    if (file.size > MAX_RECORDING_BYTES) {
      setError("Import limited to 30 MiB.");
      return;
    }
    try {
      const frames = validateRecording(
        JSON.parse(await file.text()),
        isTelemetryFrame,
      );
      if (
        session.length &&
        !window.confirm(
          "Replace the current recording? Export it first if you need to keep it.",
        )
      )
        return;
      replaceSession(frames);
      setIndex(0);
      setPlaying(false);
    } catch (err) {
      setError(err.message || "Unable to import this recording.");
    }
  }
  function exportSession() {
    downloadText(
      "DRISHTI-2.5D-recording.json",
      JSON.stringify({
        version: 1,
        recorded_at: new Date().toISOString(),
        frames: session,
      }),
    );
  }
  return (
    <main className="content-page playback-page">
      <div className="page-heading">
        <div>
          <span className="overline">SESSION REVIEW</span>
          <h1>Revisit what the system saw.</h1>
          <p>
            Replay captured telemetry locally. Recorded data never masquerades
            as live.
          </p>
        </div>
        <div className="page-actions">
          <input
            ref={input}
            type="file"
            accept=".json,application/json"
            className="sr-only"
            onChange={importFile}
            aria-label="Import recording file"
          />
          <button
            className="secondary-button"
            onClick={() => input.current.click()}
            disabled={recording}
          >
            <Icon name="upload" />
            Import
          </button>
          <button
            className="secondary-button"
            onClick={exportSession}
            disabled={!session.length}
          >
            <Icon name="download" />
            Export
          </button>
        </div>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      {!session.length ? (
        <section className="playback-empty">
          <div className="empty-icon">
            <Icon name="playback" size={34} />
          </div>
          <span className="overline">NO RECORDING LOADED</span>
          <h2>Capture a session worth revisiting.</h2>
          <p>
            Record from Live Visualization or import an exported JSON session.
            Everything stays in this browser until you choose to export it.
          </p>
          <div className="page-actions">
            <button className="primary-button" onClick={() => navigate("live")}>
              Go to live view
              <Icon name="arrow" />
            </button>
            <button
              className="secondary-button"
              onClick={() => input.current.click()}
            >
              Import recording
            </button>
          </div>
          <div className="capture-limits">
            <span>5 Hz capture</span>
            <span>30 MiB data cap</span>
            <span>600-frame limit</span>
          </div>
        </section>
      ) : (
        <>
          <div className="playback-context">
            <span className="source-tag recorded">RECORDED</span>
            <span>
              Recorded source: <b>{entry.source.toUpperCase()}</b>
            </span>
            <span className="mono">
              {session.length} frames / {duration.toFixed(1)} s
            </span>
            {recording && <span>Capture in progress</span>}
          </div>
          <div className="playback-scene">
            <SceneViewport
              frame={entry.frame}
              summary={summary}
              options={options}
              playback
              label="RECORDED TELEMETRY"
            />
          </div>
          <section className="transport" aria-label="Playback controls">
            <div className="transport-buttons">
              <button
                className="icon-button"
                aria-label="Rewind recording"
                title="Rewind to start"
                onClick={() => {
                  setIndex(0);
                  setPlaying(false);
                }}
              >
                <Icon name="back" />
              </button>
              <button
                className="primary-button"
                disabled={session.length < 2}
                onClick={() => {
                  if (index === session.length - 1) setIndex(0);
                  setPlaying((v) => !v);
                }}
              >
                <Icon name={playing ? "pause" : "playback"} />
                {playing ? "Pause" : "Play"}
              </button>
              <button
                className="icon-button"
                aria-label="Next frame"
                title="Next frame"
                disabled={session.length < 2 || index >= session.length - 1}
                onClick={() => {
                  setPlaying(false);
                  setIndex((i) => Math.min(i + 1, session.length - 1));
                }}
              >
                <Icon name="step" />
              </button>
              <label>
                Speed
                <select
                  value={speed}
                  onChange={(e) => setSpeed(+e.target.value)}
                >
                  {[0.25, 0.5, 0.75, 1, 1.5, 2].map((n) => (
                    <option key={n} value={n}>
                      {n}×
                    </option>
                  ))}
                </select>
              </label>
              <span className="mono">
                {elapsed.toFixed(1)} / {duration.toFixed(1)} s
              </span>
            </div>
            <label className="scrub-label" htmlFor="playback-timeline">
              Frame {index + 1} of {session.length}
            </label>
            <input
              id="playback-timeline"
              type="range"
              min="0"
              max={Math.max(0, session.length - 1)}
              value={index}
              onChange={(e) => {
                setPlaying(false);
                setIndex(+e.target.value);
              }}
              aria-valuetext={`Frame ${index + 1}, ${elapsed.toFixed(1)} seconds`}
            />
          </section>
        </>
      )}
    </main>
  );
}
