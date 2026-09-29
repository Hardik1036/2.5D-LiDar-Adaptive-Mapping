import React, { useState } from "react";
import { Icon } from "../components/Brand.jsx";
import { downloadText } from "../utils/session.js";
const measures = [
  ["fps", "Frame rate", "Hz", 1],
  ["latency_ms", "Update latency", "ms", 2],
  ["active_cells", "Active cells", "cells", 0],
  ["ram_mb", "RAM footprint", "MB", 2],
];
const fmt = (x, d = 1) =>
  Number.isFinite(x)
    ? x.toLocaleString("en-US", {
        maximumFractionDigits: d,
        minimumFractionDigits: d,
      })
    : "—";
export default function Analysis({
  history = [],
  summary,
  latest,
  latestFrame,
  frame: propFrame,
  status,
}) {
  const [source, setSource] = useState(
    status === "live" || status === "connected" ? "live" : "simulated",
  );
  const rows = history.filter((r) => r.source === source),
    recent = rows.slice(-10).reverse();
  const mean = (key) =>
    rows.length ? rows.reduce((n, r) => n + r[key], 0) / rows.length : null;
  const exportCsv = () => {
    const keys = [
      "at",
      "source",
      "fps",
      "latency_ms",
      "active_cells",
      "ram_mb",
      "objects",
      "point_count",
      "coarse_cells",
      "fine_cells",
      "refinement_ratio",
      "dynamic_tracks",
    ];
    downloadText(
      "DRISHTI-2.5D-telemetry-summary.csv",
      [
        keys.join(","),
        ...rows.map((r) => keys.map((k) => r[k] ?? "").join(",")),
      ].join("\n"),
      "text/csv",
    );
  };
  const resolvedFrame =
    propFrame ??
    latestFrame ??
    (latest && "current" in latest ? latest.current : latest) ??
    summary?.frame;
  const activeObjects =
    resolvedFrame?.dynamic_objects ??
    latestFrame?.dynamic_objects ??
    [];
  return (
    <main className="content-page analysis-page">
      <div className="page-heading">
        <div>
          <span className="overline">TELEMETRY REVIEW</span>
          <h1>Understand the incoming stream.</h1>
          <p>
            Bounded, in-browser diagnostics. No inferred accuracy or unverified
            model scores.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={exportCsv}
          disabled={!rows.length}
        >
          <Icon name="download" />
          Export summary
        </button>
      </div>
      <div className="analysis-toolbar">
        <div
          className="segment-control"
          role="group"
          aria-label="Analysis data source"
        >
          <button
            aria-pressed={source === "live"}
            onClick={() => setSource("live")}
          >
            Live samples
          </button>
          <button
            aria-pressed={source === "simulated"}
            onClick={() => setSource("simulated")}
          >
            Simulated samples
          </button>
        </div>
        <span className="hint">
          {rows.length} samples · up to 5 min · sampled at 1 Hz
        </span>
      </div>
      {source === "simulated" && (
        <div className="source-banner">
          <span className="amber-dot" />
          Simulation diagnostics only. These are not backend benchmarks.
        </div>
      )}
      <div className="analysis-metrics">
        {measures.map(([key, label, unit, d]) => (
          <section className="data-card" key={key}>
            <h2>{label}</h2>
            <div className="large-value">
              {fmt(rows.at(-1)?.[key], d)}
              <small>{unit}</small>
            </div>
            <div className="metric-comparison">
              <span>
                Mean <b>{fmt(mean(key), d)}</b>
              </span>
              <span>
                Peak{" "}
                <b>
                  {fmt(
                    rows.length ? Math.max(...rows.map((r) => r[key])) : null,
                    d,
                  )}
                </b>
              </span>
            </div>
          </section>
        ))}
      </div>
      <div className="analysis-columns">
        <section className="surface-card">
          <div className="card-heading-row">
            <div>
              <span className="overline">CURRENT FRAME</span>
              <h2>Terrain composition</h2>
            </div>
            <span className={`source-tag ${status}`}>
              {status.toUpperCase()}
            </span>
          </div>
          <p className="hint">
            All received cells, independent of viewport filters.
          </p>
          <div className="terrain-composition">
            {[
              ["Safe ground", "0–50", "#2EA043"],
              ["Caution", "51–180", "#D29922"],
              ["Hazard", "181–255", "#F85149"],
            ].map(([label, range, color], i) => (
              <div className="composition-row" key={label}>
                <div>
                  <i style={{ background: color }} />
                  <strong>{label}</strong>
                  <span className="mono">{range}</span>
                  <b>{summary?.bands[i].toLocaleString() ?? "—"}</b>
                </div>
                <div className="bar-track">
                  <span
                    style={{
                      background: color,
                      width: `${summary?.total ? (summary.bands[i] / summary.total) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="frame-total">
            <span>Received cell array</span>
            <b>{summary?.total.toLocaleString() ?? "—"} cells</b>
          </div>
        </section>
        <section className="surface-card">
          <div className="card-heading-row">
            <div>
              <span className="overline">CURRENT FRAME</span>
              <h2>Tracked objects</h2>
            </div>
            <span className="mono">{activeObjects.length} objects</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Class</th>
                  <th>Position</th>
                  <th>Velocity</th>
                  <th>Speed</th>
                  <th>Heading</th>
                </tr>
              </thead>
              <tbody>
                {activeObjects.slice(0, 20).map((obj) => {
                  const posX = Number(obj.x ?? 0).toFixed(2);
                  const posY = Number(obj.y ?? 0).toFixed(2);
                  const velX = Number(obj.vx || 0).toFixed(2);
                  const velY = Number(obj.vy || 0).toFixed(2);
                  const speedText = obj.stationary
                    ? "0.00 m/s (Stationary)"
                    : `${(Number(obj.speed) || 0).toFixed(2)} m/s`;
                  const heading = `${(
                    (Number(obj.heading) || 0) *
                    (180 / Math.PI)
                  ).toFixed(1)}°`;

                  return (
                    <tr key={obj.id}>
                      <td>
                        <b>{obj.id}</b>
                      </td>
                      <td>{obj.class ?? "obstacle"}</td>
                      <td className="mono">({posX}, {posY})</td>
                      <td className="mono">vx: {velX}, vy: {velY}</td>
                      <td className="mono">{speedText}</td>
                      <td className="mono">{heading}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!activeObjects.length && (
            <div
              className="empty-inline"
              style={{
                padding: "24px 16px",
                textAlign: "center",
                color: "#94A3B8",
              }}
            >
              <strong style={{ color: "#E2E8F0", letterSpacing: "1px" }}>
                NO ACTIVE DYNAMIC TRACKS IN SENSOR FOV
              </strong>
              <p
                style={{
                  marginTop: "4px",
                  fontSize: "13px",
                  color: "var(--muted)",
                }}
              >
                Awaiting dynamic objects from LiDAR perception stream.
              </p>
            </div>
          )}
          {activeObjects.length > 20 && (
            <p className="hint">
              Showing the first 20 of {activeObjects.length} objects.
            </p>
          )}
        </section>
      </div>
      <section className="surface-card">
        <div className="card-heading-row">
          <div>
            <span className="overline">RECEIVED OBSERVATIONS</span>
            <h2>Recent telemetry</h2>
          </div>
          <span className="hint">Latest 10 selected-source samples</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Received at</th>
                <th>Frame rate, Hz</th>
                <th>Latency, ms</th>
                <th>Active cells</th>
                <th>RAM, MB</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r) => (
                <tr key={r.at}>
                  <td>
                    {new Date(r.at).toLocaleTimeString("en-GB", {
                      hour12: false,
                    })}
                  </td>
                  <td>{fmt(r.fps, 1)}</td>
                  <td>{fmt(r.latency_ms, 2)}</td>
                  <td>{fmt(r.active_cells, 0)}</td>
                  <td>{fmt(r.ram_mb, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!rows.length && (
          <div className="empty-inline">
            No {source} samples yet.{" "}
            {source === "live"
              ? "Connect a backend that sends valid telemetry."
              : "The fallback starts after three seconds without a valid frame."}
          </div>
        )}
      </section>
    </main>
  );
}
