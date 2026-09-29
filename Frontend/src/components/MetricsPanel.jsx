import React from "react";
import { DrishtiSymbol } from "./Brand.jsx";
import { MODE_SEMANTIC } from "../three/Scene.js";
import AccuracyMeter from "./AccuracyMeter.jsx";
const format = (n, decimals = 0) =>
  typeof n === "number"
    ? n.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
    : "—";
const bands = [
  ["Safe ground", "0–50", "#2EA043"],
  ["Caution", "51–180", "#D29922"],
  ["Hazard", "181–255", "#F85149"],
];
export default function MetricsPanel({ summary, status, options }) {
  const stats = summary?.stats;
  const height = options.mode !== MODE_SEMANTIC && options.colorBy === "height";
  return (
    <aside className="panel metrics-panel" aria-label="Live Metrics">
      <div className="panel-title">
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <DrishtiSymbol size={20} />
          <h2>Live Metrics</h2>
        </div>
        <span className="section-mark">FEED</span>
      </div>
      <p className="source-note">
        {status === "live" || status === "connected" ? (
          <span>
            Telemetry Source:{" "}
            <strong style={{ color: "#3fb950" }}>
              Drishti-2.5D Telemetry Core
            </strong>
          </span>
        ) : status === "simulated" ? (
          "Simulated telemetry · awaiting backend"
        ) : (
          "Disconnected · retrying with exponential backoff"
        )}
      </p>
      <div className="accuracy-meter-wrapper" style={{ marginBottom: "20px" }}>
        <AccuracyMeter
          value={stats?.accuracy ?? stats?.tracking_accuracy ?? null}
          label="Perception Fidelity"
        />
      </div>
      <div className="metrics-grid">
        {[
          ["Frame rate", stats?.fps, 1, "Hz"],
          ["Update latency", stats?.latency_ms, 2, "ms"],
          ["Active cells", stats?.active_cells, 0, "cells"],
          ["RAM footprint", stats?.ram_mb, 2, "MB"],
        ].map(([label, value, decimals, unit]) => (
          <div className="metric" key={label}>
            <span>{label}</span>
            <div>
              <strong>{format(value, decimals)}</strong>
              <small>{unit}</small>
            </div>
          </div>
        ))}
      </div>
      {(stats?.point_count !== undefined ||
        stats?.coarse_cells !== undefined) && (
          <section className="metric-section">
            <div className="label-row">
              <h3>2.5D Partitioning</h3>
              <span className="section-mark">QUADTREE</span>
            </div>
            <div className="metrics-grid">
              <div className="metric">
                <span>Points</span>
                <div>
                  <strong>{format(stats?.point_count, 0)}</strong>
                  <small>pts</small>
                </div>
              </div>
              <div className="metric">
                <span>Coarse / Fine</span>
                <div>
                  <strong>
                    {format(stats?.coarse_cells, 0)} / {format(stats?.fine_cells, 0)}
                  </strong>
                </div>
              </div>
              <div className="metric">
                <span>Refine ratio</span>
                <div>
                  <strong>
                    {typeof stats?.refinement_ratio === "number"
                      ? `${(stats.refinement_ratio * 100).toFixed(1)}%`
                      : "—"}
                  </strong>
                </div>
              </div>
              <div className="metric">
                <span>Dynamic tracks</span>
                <div>
                  <strong>
                    {format(stats?.dynamic_tracks ?? stats?.active_tracks, 0)}
                  </strong>
                  <small>tracks</small>
                </div>
              </div>
            </div>
          </section>
        )}
      <section className="reference-badge">
        <h3>Reference benchmark</h3>
        <p>
          RAM Footprint: <b>&lt; 8.0 MB</b>
        </p>
        <span>Embedded Edge Budget: &lt; 10 MB target</span>
      </section>
      <section className="metric-section">
        <h3>{height ? "Elevation legend" : "Terrain cost legend"}</h3>
        {height ? (
          <>
            <div className="elevation-scale" aria-hidden="true">
              {["#3e8aeb", "#3eb9eb", "#3eebcb", "#98eb3e", "#eb863e"].map(
                (c) => (
                  <i key={c} style={{ background: c }} />
                ),
              )}
            </div>
            <div className="range-labels mono">
              <span>{options.minHeight.toFixed(1)} m</span>
              <span>{options.maxHeight.toFixed(1)} m</span>
            </div>
            <p className="hint">Minimum elevation (z_min)</p>
          </>
        ) : (
          bands.map(([label, range, color]) => (
            <div className="legend-row" key={label}>
              <i style={{ background: color }} />
              <span>{label}</span>
              <span className="mono">{range}</span>
            </div>
          ))
        )}
      </section>
      <section className="metric-section">
        <div className="label-row">
          <h3>Classification</h3>
          <span className="section-mark">ALL CELLS</span>
        </div>
        {bands.map(([label, , color], i) => (
          <div className="classification-row" key={label}>
            <div>
              <span>{label}</span>
              <strong>{summary ? format(summary.bands[i]) : "—"}</strong>
            </div>
            <div className="bar-track">
              <span
                style={{
                  width: `${summary?.total ? (summary.bands[i] / summary.total) * 100 : 0}%`,
                  background: color,
                }}
              />
            </div>
          </div>
        ))}
        <p className="hint">
          Cost-band counts from the full received frame; unaffected by view
          filters.
        </p>
      </section>
    </aside>
  );
}
