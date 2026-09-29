# SIH26053 — Three.js LiDAR Frontend

Frontend-only replacement. React 18 + Vite 6 + Three.js. No backend, ML model, downloaded dataset, external models, fonts, or image assets.

## Run

Use Node.js 20.19+ or 22+ and npm.

```sh
npm ci
npm run dev
```

Open **http://localhost:5173** (use `localhost`, not `127.0.0.1`, to select the local backend).

```sh
npm test
npm run build
npm run preview
```

The three-second fallback means the demo works without a backend. Reloading the page reconnects automatically. Desktop sidebars scroll independently to keep the scene visible; mobile stacks scene, controls, and metrics.

## Backend handoff — one file

`src/services/telemetryService.js` owns the URL, validation, connection, reconnection, stale-frame watchdog, and internal simulation. `connectTelemetry(onFrame, onStatusChange)` returns a cleanup function. No renderer changes are required for a feed matching the documented contract.

- `localhost` → `ws://localhost:8765`
- Other hostnames → `wss://steadfast-liberation-production.up.railway.app`
- Retry delay: 1, 2, 4, 8, then 10 seconds maximum.
- Fallback begins after three seconds without valid telemetry, even if the socket opens but sends nothing.
- A fresh, valid frame stops local fallback immediately. Silent live streams fall back after three seconds and reconnect.
- Status callbacks: `connecting`, `live`, `offline`, `simulated`. Fallback keeps `simulated` visible during retries. A backend frame explicitly marked `SIMULATED` also displays simulated.
- Malformed JSON, missing fields, non-finite values, invalid dimensions, and duplicate object IDs are ignored. Legitimate empty arrays and zero metrics are accepted.
- Cleanup stops sockets, callbacks, timers, and retries. React StrictMode is supported.

### Required JSON contract

```json
{
  "timestamp": 1788868200000,
  "frame_id": 42,
  "system_status": "LIVE",
  "system_stats": {
    "fps": 25,
    "latency_ms": 12.4,
    "active_cells": 1,
    "ram_mb": 0.1
  },
  "cells": [
    { "x": 0, "y": 0, "size": 1, "cost": 25, "z_min": 0, "z_max": 0.2 }
  ],
  "dynamic_objects": [
    {
      "id": "p1",
      "class": "pedestrian",
      "x": 2,
      "y": 4,
      "vx": 0,
      "vy": 1.2,
      "speed": 1.2,
      "heading": 1.5707963268
    }
  ]
}
```

**Confirm these conventions with your teammate:** numeric Unix timestamp in milliseconds; heading in radians counterclockwise from +X; coordinates/elevations in metres; velocities and speed in m/s; string class labels. If different, normalize them inside the service before validation/emission. The supplied schema alone does not define heading or timestamp units.

Backend X/Y is the horizontal plane, Z is elevation. Three.js receives `(x, z, -y)`. Each cell X/Y is assumed to be its center. If the backend sends lower-left corners, normalize centers in the service. Actors use Z=0 because the object schema has no elevation. No collision avoidance, inference, or motion-planning claims are made.

## Rendering and controls

- **Elevation Grid:** flat instanced tiles at `z_min`. Footprints are `size × size` at scale 1.0. Capacity grows rather than silently truncating large frames. Shader edges expose adaptive cell boundaries without per-cell draw calls.
- **Classification:** flat tiles at Z=0, colored by traversal-cost band.
- **Point Cloud:** TWO samples per cell, at `z_min` and `z_max`. This is explicitly **not raw LiDAR**: there is no raw-points field in the supplied schema.
- **Scale:** changes tile footprint or point size, not underlying telemetry values.
- **Height sliders:** filter by `z_min`; color-by-height is normalized to this range.
- **Color:** traversal cost or elevation. Classification locks cost coloring by design.
- **Reference layers:** ground plane, grid, axes; camera reset by button or R. Orbit, wheel zoom, and right-drag pan work through OrbitControls.
- **Objects:** vehicle boxes, capsule/box humanoids; unrecognized classes use a neutral box. IDs persist across frames; absent IDs are removed. Heading rotates the avatar. Speed selects idle (<0.2), walk (0.2–1.8), or run (>1.8 m/s) for humanoids. Reduced-motion disables cosmetic limb/bob animation.
- **Hazards:** translucent 0.35-opacity ground wedges follow velocity, using a two-second projection with a one-metre visual minimum. These are illustrative velocity envelopes, not collision predictions.
- High-rate frames go directly to the scene; the React metric summary updates at 5 Hz. Rendering resources and event listeners are disposed on unmount.

## Honest metrics

The four metrics display `system_stats` without substituting browser render FPS or hardcoded live values. Classification counts derive from `cells`, independent of view filters. `active_cells` is shown as reported by the backend, even if it differs from `cells.length`.

During **local simulation**, FPS is timer cadence (capped at target 25 Hz), latency is local frame generation time, and RAM is a numeric cell-payload estimate—not browser/process/GPU memory. These must not be used as algorithm benchmarks. Software rendering in a sandbox can reduce actual cadence substantially.

The requested **“RAM Footprint: 0.10 MB | 99.48% Memory Reduction”** is clearly labeled a static reference target, not verified or measured performance.

## Cleanup audit

This is a new replacement source tree, not a patch layered onto the old application. The application imports only the current page, component, hook, service, and utility modules. Its complete import graph was checked before packaging.

Removed from the replacement: old App and styles; TopBar; RightPanel; Footer; PhaseSidebar; PhaseBottomBar; ViewportHud; useLidarSocket; old scene renderers; syntheticFallback; old README; old build output. No new source imports those files.

The specifically requested `phase1.js`–`phase4.js`, `syntheticData.js`, `MainCanvas.jsx`, `ActivityLog.jsx`, `RadarMinimap.jsx`, and `BottomBar.jsx` were absent from the uploaded source tree. They are also absent from this replacement.

Excluded: `node_modules`, `dist`, `.DS_Store`, `__MACOSX`, legacy panels, old phase UI, unused dependencies, screenshots, and test-only browser mocks. Only React, React DOM, and Three.js are runtime dependencies. Vite and its React plugin are build-time dependencies.

## Validation performed

- Fourteen automated service/session lifecycle and validation tests passed (`npm test`, also runnable directly with Node).
- An offline production bundle built successfully using the ZIP's exact React/Three.js dependencies and the available esbuild tool.
- Chromium: actual WebGL rendering, all modes, color switch, sliders, reference toggles, camera actions, empty-height state, and mobile overflow checked. No runtime exceptions in this interaction test.
- Desktop and 390px mobile layouts visually reviewed.
- **Not verified:** `npm ci` / the standard Vite build in this sandbox (npm registry DNS/network access was unavailable); real teammate WebSocket integration; backend/ML performance or memory claims.

The optional separately supplied standalone HTML preview bundles the same frontend. It is not a backend and has no test socket mock. It can fall back locally, but served development via localhost is recommended for backend integration.

# Multi-page edition — welcome, analyze, record, replay

This updated edition opens with a **three-slide project briefing**, then provides distinct hash-routed pages: `#welcome`, `#live`, `#analysis`, and `#playback`. Browser back/forward navigation works without adding a routing dependency. Settings is an accessible native modal, not a placeholder page.

## Welcome and branding

- Three manually navigated slides: Mission, Architecture, and Ready to Operate.
- Original gold/blue vector project emblem; **not an official DRDO logo**.
- Local Three.js cartographic illustration and restrained floating explanatory labels.
- Motion pause control and OS reduced-motion support. The welcome terrain is explicitly illustrative, not sensor data.
- Direct entry to the live workspace; clicking the project mark returns to the briefing.

## Analysis

- Latest/mean/peak of reported system statistics, using an explicitly selected live or simulated sample set; sources are not pooled.
- Rolling buffer: 300 samples, at most one sample per second. This covers roughly five minutes of uninterrupted reception; gaps may extend the wall-clock span.
- Full-frame cost-band composition, the first 20 current tracked objects, and latest 10 selected-source telemetry samples.
- CSV export of the selected-source diagnostic buffer, entirely in the browser.
- No inferred model accuracy, fabricated sensor range, or unsupported ML statistics.

## Recording and playback

- Record/stop in Live Visualization. Sampling is capped at 5 Hz to keep capture bounded; this is intentionally not a full 25 Hz archive.
- Limits: 600 frames or a 30 MiB **serialized-payload budget**, whichever is reached first. This is not a guarantee of total JavaScript heap usage; object overhead and temporary export strings use additional memory.
- Recordings contain elapsed arrival time, source labels, and full schema frames. They preserve the live/simulated distinction when the source changes.
- Local JSON import validates format version, ordered timeline, frame schema, source labels, file-size budget, and frame limit.
- Play/pause, rewind, timeline scrubbing, 0.5×/1×/2× playback, and JSON export are functional.
- The playback scene uses recorded frames, never the latest live frame. The header's FEED indicator refers to the separately running connection; the scene is marked RECORDED.
- Imports/exports do not contact a server. Recordings remain in browser memory until exported. Refreshing or closing the page loses unexported recordings. New capture/import asks before replacing an existing recording.
- Source labels on imported files are metadata, not cryptographically verified evidence.

## Settings and resource use

Pixel ratio changes renderer resolution (1×, 1.5×, 2×, capped by device pixel ratio). Humanoid gait can be paused; actual telemetry positions continue updating. Scene preference reset is functional. Preferences are session-only. Navigating away disposes the active Three.js scene; welcome and live/playback renderers do not accumulate.

## Updated validation

**14 automated tests passed**, covering the service lifecycle plus recording validation and cost-band boundaries. Chromium interaction testing passed: three slides, routes, recording, JSON export/import, timed replay, rewind, CSV export, settings, camera reset, and all four pages at 390px. Desktop, mobile, playback, and modal screenshots were reviewed. Production code bundled offline with esbuild and the exact supplied React/Three.js versions.

The real teammate backend has still **not** been contacted or verified. Standard npm/Vite installation/build remains unverified in this sandbox due to the earlier registry network restriction. Run `npm ci`, `npm test`, and `npm run build` on your development machine.

The previous single-dashboard ZIP remains unchanged; use this multi-page edition instead when you want the welcome and session workflow.
