import React, { useState, useRef } from "react";
import { DrishtiSymbol, Icon } from "./Brand.jsx";
import {
  MODES,
  MODE_POINTS,
  MODE_ADAPTIVE,
  MODE_SEMANTIC,
  MODE_PERCEPTION,
} from "../three/Scene.js";
import {
  setLivePlaybackSpeed,
  getLivePlaybackSpeed,
  setLivePaused,
  isLivePaused,
  stepLiveFrame,
  sendCommand,
  setDatasetMode,
  telemetryService,
} from "../services/telemetryService.js";
const MODE_HINT = {
  [MODE_POINTS]:
    "Cell elevation samples plus tracked-object returns. Object meshes stay hidden.",
  [MODE_ADAPTIVE]:
    "Adaptive tiles at each cell’s minimum elevation. Moving objects re-cost the cells they occupy.",
  [MODE_SEMANTIC]:
    "Flat cost-band map with classified moving objects and their 2 s hazard projection.",
  [MODE_PERCEPTION]:
    "Vehicle-centric view: detected structures, vegetation, boundaries, depressions and tracked objects.",
};
export default function SceneControls({
  options: o,
  setOptions,
  onReset,
  isPaused: propIsPaused,
  onTogglePause,
  onStepFrame,
  playbackSpeed: propPlaybackSpeed,
  onSpeedChange,
}) {
  const change = (key, value) => setOptions((p) => ({ ...p, [key]: value }));

  const [activeDataset, setActiveDataset] = useState("static"); // "static" | "dynamic"
  const datasetDebounceTimer = useRef(null);

  const handleSelectDataset = (mode) => {
    setActiveDataset(mode);
    if (datasetDebounceTimer.current) {
      clearTimeout(datasetDebounceTimer.current);
    }
    datasetDebounceTimer.current = setTimeout(() => {
      setDatasetMode(mode);
    }, 300);
  };

  const isPaused = propIsPaused !== undefined ? propIsPaused : isLivePaused();
  const currentSpeed =
    propPlaybackSpeed !== undefined
      ? propPlaybackSpeed
      : o.playbackSpeed || getLivePlaybackSpeed() || 1.0;

  const handleTogglePause = () => {
    if (onTogglePause) {
      onTogglePause();
    } else {
      const next = !isLivePaused();
      setLivePaused(next);
    }
  };

  const handleStepFrame = () => {
    if (onStepFrame) {
      onStepFrame();
    } else {
      stepLiveFrame();
    }
  };

  const handleSpeedChange = (speed) => {
    const s = Math.max(0.1, Math.min(5.0, Number(speed) || 1.0));
    change("playbackSpeed", s);
    setLivePlaybackSpeed(s);
    if (onSpeedChange) {
      onSpeedChange(s);
    }
    // Transmit command to backend so the server throttles accordingly
    if (telemetryService && typeof telemetryService.sendMessage === "function") {
      telemetryService.sendMessage({
        type: "set_speed",
        speed: s,
      });
    }
  };

  return (
    <aside className="panel controls-panel" aria-label="Scene Controls">
      <div className="panel-title">
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <DrishtiSymbol size={20} />
          <h2>Scene Controls</h2>
        </div>
        <span className="section-mark">VIEW</span>
      </div>
      <section className="control-section">
        <h3>View mode</h3>
        <div className="view-modes" role="group" aria-label="View mode">
          {MODES.map((mode, i) => (
            <button
              key={mode}
              aria-pressed={o.mode === mode}
              onClick={() => change("mode", mode)}
            >
              <span className="mode-symbol" aria-hidden="true">
                {i === 0 ? (
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                  >
                    {[3, 8, 13].flatMap((x) =>
                      [3, 8, 13].map((y) => (
                        <circle key={`${x}-${y}`} cx={x} cy={y} r="1" />
                      )),
                    )}
                  </svg>
                ) : i === 3 ? (
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  >
                    <circle cx="8" cy="8" r="6.3" />
                    <circle cx="8" cy="8" r="2.6" />
                    <path d="M8 8l4.4-3.2" />
                  </svg>
                ) : (
                  ["", "▦", "▤"][i]
                )}
              </span>
              {mode}
              <span className="radio-dot" aria-hidden="true" />
            </button>
          ))}
        </div>
        <p className="hint">{MODE_HINT[o.mode]}</p>
        <p className="hint swipe-hint">
          Swipe or scroll horizontally over the scene to change mode.
        </p>
      </section>
      <section className="control-section playback-controls-card">
        <div className="playback-header-row">
          <div className="playback-title-wrap">
            <label htmlFor="playback-speed">Playback speed</label>
            <span className={`playback-live-pill ${isPaused ? "paused" : "live"}`}>
              <span className="live-dot" />
              {isPaused ? "PAUSED" : "LIVE"}
            </span>
          </div>
          <output htmlFor="playback-speed" className="speed-badge">
            {currentSpeed.toFixed(2).replace(/\.?0+$/, "")}×
          </output>
        </div>

        {/* Transport controls: Pause / Play, Next Frame */}
        <div className="playback-transport-bar" role="group" aria-label="Playback stream transport">
          <button
            type="button"
            className={`transport-btn play-pause-btn ${isPaused ? "is-paused" : "is-playing"}`}
            onClick={handleTogglePause}
            aria-label={isPaused ? "Play stream" : "Pause stream"}
            title={isPaused ? "Resume real-time playback" : "Pause live frame processing"}
          >
            <Icon name={isPaused ? "playback" : "pause"} size={16} />
            <span>{isPaused ? "Play" : "Pause"}</span>
          </button>

          <button
            type="button"
            className="transport-btn step-btn"
            onClick={handleStepFrame}
            aria-label="Next frame"
            title="Step forward to next frame / sweep"
          >
            <Icon name="step" size={16} />
            <span>Next</span>
          </button>
        </div>

        {/* Speed range slider supporting slow speeds down to 0.1x */}
        <input
          id="playback-speed"
          type="range"
          min="0.1"
          max="2"
          step="0.05"
          value={currentSpeed}
          onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
          aria-label="Playback speed slider"
        />

        {/* Presets including slow speeds (0.25x, 0.5x, 0.75x) as requested */}
        <div className="speed-presets-grid" role="group" aria-label="Speed presets">
          {[0.25, 0.5, 0.75, 1, 1.5, 2].map((s) => (
            <button
              key={s}
              type="button"
              className={`preset-btn ${Math.abs(currentSpeed - s) < 0.02 ? "active" : ""}`}
              onClick={() => handleSpeedChange(s)}
            >
              {s}×
            </button>
          ))}
        </div>
      </section>
      {/* Dataset Sequence Selector */}
      <section className="control-section">
        <div className="mt-4 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold mb-2">
            Dataset Sequence
          </div>

          {/* Strict 2-Button Toggle */}
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleSelectDataset("static")}
              className={`py-2 px-3 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                activeDataset === "static"
                  ? "bg-[#238636] text-white shadow-lg shadow-emerald-600/25 border border-emerald-400/40"
                  : "bg-slate-800/80 text-slate-400 hover:text-white hover:bg-slate-700/60 border border-slate-700/50"
              }`}
            >
              STATIC
            </button>

            <button
              type="button"
              onClick={() => handleSelectDataset("dynamic")}
              className={`py-2 px-3 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                activeDataset === "dynamic"
                  ? "bg-[#388BFD] text-white shadow-lg shadow-blue-600/25 border border-blue-400/40"
                  : "bg-slate-800/80 text-slate-400 hover:text-white hover:bg-slate-700/60 border border-slate-700/50"
              }`}
            >
              DYNAMIC
            </button>
          </div>

          {/* Status Feedback */}
          <p className="mt-2 text-[10px] font-mono text-slate-400 tracking-tight leading-relaxed">
            {activeDataset === "static"
              ? "Streaming clean baseline corridor from Backend/data/static_corridor/."
              : "Streaming continuous multi-object dynamic tracking from Backend/data/dynamic_corridor/."}
          </p>
        </div>
      </section>
      <section className="control-section">
        <div className="label-row">
          <label htmlFor="cell-size">Point / cell scale</label>
          <output htmlFor="cell-size">{o.size.toFixed(2)}×</output>
        </div>
        <input
          id="cell-size"
          type="range"
          min="0.4"
          max="1.5"
          step="0.05"
          value={o.size}
          onChange={(e) => change("size", +e.target.value)}
        />
        <div className="range-labels">
          <span>0.4×</span>
          <span>1.5×</span>
        </div>
        <div className="label-row height-label">
          <h3>Height range</h3>
          <span className="mono">m</span>
        </div>
        <label className="range-name" htmlFor="height-min">
          Minimum <output>{o.minHeight.toFixed(1)} m</output>
        </label>
        <input
          id="height-min"
          type="range"
          min="-5.0"
          max="5.0"
          step="0.1"
          value={o.minHeight}
          onChange={(e) =>
            change("minHeight", Math.min(parseFloat(e.target.value), o.maxHeight - 0.1))
          }
        />
        <div className="range-labels">
          <span>-5.0 m</span>
          <span>5.0 m</span>
        </div>
        <label className="range-name" htmlFor="height-max">
          Maximum <output>{o.maxHeight.toFixed(1)} m</output>
        </label>
        <input
          id="height-max"
          type="range"
          min="-5.0"
          max="10.0"
          step="0.1"
          value={o.maxHeight}
          onChange={(e) =>
            change("maxHeight", Math.max(parseFloat(e.target.value), o.minHeight + 0.1))
          }
        />
        <div className="range-labels">
          <span>-5.0 m</span>
          <span>10.0 m</span>
        </div>
        <p className="hint">Filters cells by z_min.</p>
      </section>
      <section className="control-section">
        <label htmlFor="color-by">Color by</label>
        <select
          id="color-by"
          value={o.mode === MODE_SEMANTIC ? "cost" : o.colorBy}
          disabled={o.mode === MODE_SEMANTIC}
          onChange={(e) => change("colorBy", e.target.value)}
        >
          <option value="cost">Traversal cost</option>
          <option value="height">Minimum elevation</option>
        </select>
        {o.mode === MODE_SEMANTIC && (
          <p className="hint">Semantic hazard view uses cost bands.</p>
        )}
      </section>
      <section className="control-section">
        <h3>Reference layers</h3>
        {[
          ["ground", "Ground plane"],
          ["grid", "Grid"],
          ["axes", "Coordinate axes"],
        ].map(([key, label]) => (
          <label className="check-row" key={key}>
            <span>{label}</span>
            <input
              type="checkbox"
              checked={o[key]}
              onChange={(e) => change(key, e.target.checked)}
            />
          </label>
        ))}
        {o.axes && <p className="hint">Red: X · Blue: −Y · Green: Z up</p>}
      </section>
      <button className="reset-button" onClick={onReset}>
        Reset camera <kbd>R</kbd>
      </button>
    </aside>
  );
}
