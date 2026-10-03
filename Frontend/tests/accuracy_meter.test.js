import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import esbuild from "esbuild";
import React from "react";
import { renderToString } from "react-dom/server";
import { summarizeFrame } from "../src/utils/session.js";

// Transform AccuracyMeter.jsx for pure Node test runner
const meterSrc = fs
  .readFileSync(new URL("../src/components/AccuracyMeter.jsx", import.meta.url), "utf8")
  .replace(/import\s+React\s+from\s+["']react["'];?/, "")
  .replace("export default function AccuracyMeter", "return function AccuracyMeter");

const transformed = esbuild.transformSync(meterSrc, { loader: "jsx" }).code;
const AccuracyMeter = new Function("React", transformed)(React);

function resolveValidationMetrics(summary) {
  const valSource =
    summary?.frame?.model_validation ??
    summary?.frame?.system_stats?.model_validation ??
    summary?.stats?.model_validation ??
    summary?.frame?.model_status?.model_validation ??
    null;

  let validationMap = null;
  if (typeof valSource === "number") {
    validationMap = valSource;
  } else if (typeof valSource?.mAP === "number") {
    validationMap = valSource.mAP;
  } else if (typeof summary?.frame?.system_stats?.model_validation?.mAP === "number") {
    validationMap = summary.frame.system_stats.model_validation.mAP;
  }

  const isAvailable = validationMap !== null && validationMap !== undefined;
  return {
    value: validationMap,
    label: "POINTPILLARS mAP",
    status: isAvailable ? "OFFLINE VALIDATION" : "NO VALIDATION DATA",
    subtitle: "NuScenes validation · 81 samples",
  };
}

test("AccuracyMeter renders 9.7%, POINTPILLARS mAP, and OFFLINE VALIDATION for mAP=9.73", () => {
  const html = renderToString(
    React.createElement(AccuracyMeter, {
      value: 9.73,
    })
  );

  assert.ok(html.includes("9.7%"), "Must display 9.7% (formatted 9.73)");
  assert.ok(html.includes("POINTPILLARS mAP"), "Must display POINTPILLARS mAP label");
  assert.ok(html.includes("OFFLINE VALIDATION"), "Must display OFFLINE VALIDATION status");
  assert.ok(html.includes("NuScenes validation · 81 samples"), "Must display NuScenes sample subtitle");
  assert.ok(!html.includes("STANDBY"), "Must NOT display STANDBY when validation metric is available");
  assert.ok(!html.includes("RMSE"), "Must NOT display misleading RMSE subtitle");
});

test("AccuracyMeter with missing validation renders --.-% and NO VALIDATION DATA", () => {
  const htmlNull = renderToString(
    React.createElement(AccuracyMeter, {
      value: null,
    })
  );

  assert.ok(htmlNull.includes("--.-%"), "Must display --.-% when value is null");
  assert.ok(htmlNull.includes("NO VALIDATION DATA"), "Must display NO VALIDATION DATA");
  assert.ok(htmlNull.includes("POINTPILLARS mAP"), "Must display POINTPILLARS mAP label");
  assert.ok(!htmlNull.includes("STANDBY"), "Must NOT display STANDBY");

  const htmlUndefined = renderToString(
    React.createElement(AccuracyMeter, {})
  );
  assert.ok(htmlUndefined.includes("--.-%"), "Must display --.-% when value is undefined");
  assert.ok(htmlUndefined.includes("NO VALIDATION DATA"), "Must display NO VALIDATION DATA");
});

test("MetricsPanel telemetry wiring extracts model_validation and renders PointPillars mAP", () => {
  const rawFrame = {
    frame_id: 100,
    timestamp: 1725612345000,
    system_status: "ALL_SYSTEMS_NOMINAL",
    mode: "2.5D ADAPTIVE",
    model_validation: {
      model_name: "PointPillars",
      dataset: "NuScenes validation",
      samples: 81,
      mAP: 9.73,
      NDS: 14.90,
      live_ground_truth: "NOT AVAILABLE",
    },
    system_stats: {
      fps: 25.0,
      latency_ms: 12.0,
      active_cells: 50,
      ram_mb: 2.1,
      tracking_accuracy: null,
      ground_truth_status: "NOT AVAILABLE",
      model_validation: {
        mAP: 9.73,
        NDS: 14.90,
      },
    },
    cells: [],
    dynamic_objects: [],
  };

  const summary = summarizeFrame(rawFrame);
  assert.equal(summary.accuracy, null, "Live tracking accuracy must remain null without ground truth");

  const props = resolveValidationMetrics(summary);
  assert.equal(props.value, 9.73);
  assert.equal(props.label, "POINTPILLARS mAP");
  assert.equal(props.status, "OFFLINE VALIDATION");
  assert.equal(props.subtitle, "NuScenes validation · 81 samples");

  const html = renderToString(React.createElement(AccuracyMeter, props));
  assert.ok(html.includes("9.7%"));
  assert.ok(html.includes("POINTPILLARS mAP"));
  assert.ok(html.includes("OFFLINE VALIDATION"));
  assert.ok(html.includes("NuScenes validation · 81 samples"));
  assert.ok(!html.includes("STANDBY"));
});

test("MetricsPanel telemetry wiring resolves system_stats.model_validation.mAP robustly", () => {
  const rawFrame = {
    frame_id: 101,
    timestamp: 1725612346000,
    system_status: "ALL_SYSTEMS_NOMINAL",
    system_stats: {
      fps: 25.0,
      latency_ms: 12.0,
      tracking_accuracy: null,
      ground_truth_status: "NOT AVAILABLE",
      model_validation: {
        mAP: 9.73,
      },
    },
    cells: [],
    dynamic_objects: [],
  };

  const summary = summarizeFrame(rawFrame);
  const props = resolveValidationMetrics(summary);
  assert.equal(props.value, 9.73);
  assert.equal(props.status, "OFFLINE VALIDATION");

  const html = renderToString(React.createElement(AccuracyMeter, props));
  assert.ok(html.includes("9.7%"));
  assert.ok(html.includes("POINTPILLARS mAP"));
  assert.ok(html.includes("OFFLINE VALIDATION"));
});

test("MetricsPanel with missing validation and live tracking_accuracy=null does NOT fabricate accuracy", () => {
  const rawFrame = {
    frame_id: 102,
    timestamp: 1725612347000,
    system_status: "ALL_SYSTEMS_NOMINAL",
    system_stats: {
      fps: 25.0,
      latency_ms: 12.0,
      tracking_accuracy: null,
      ground_truth_status: "NOT AVAILABLE",
    },
    cells: [],
    dynamic_objects: [],
  };

  const summary = summarizeFrame(rawFrame);
  assert.equal(summary.accuracy, null);

  const props = resolveValidationMetrics(summary);
  assert.equal(props.value, null);
  assert.equal(props.status, "NO VALIDATION DATA");

  const html = renderToString(React.createElement(AccuracyMeter, props));
  assert.ok(html.includes("--.-%"), "Must show --.-% when validation data missing");
  assert.ok(html.includes("NO VALIDATION DATA"), "Must show NO VALIDATION DATA");
  assert.ok(!html.includes("9.7%"), "Must NOT fabricate a percentage");
  assert.ok(!html.includes("STANDBY"));
});
