import React, { useEffect, useRef, useState } from "react";
import {
  LidarScene,
  MODES,
  MODE_POINTS,
  MODE_ADAPTIVE,
  MODE_SEMANTIC,
  MODE_PERCEPTION,
} from "../three/Scene.js";
import { Icon } from "./Brand.jsx";

export default function SceneViewport({
  frameRef,
  frame,
  summary,
  options,
  label = "LIVE VISUALIZATION",
  playback = false,
  onModeChange,
}) {
  const mount = useRef(null),
    engine = useRef(null),
    [error, setError] = useState("");

  const data = useRef({ frameRef, frame });
  data.current = { frameRef, frame };

  useEffect(() => {
    let poll;
    try {
      const instance = new LidarScene(mount.current, setError);
      engine.current = instance;
      let previous = null;
      const loop = () => {
        const d = data.current;
        const currentFrame = d.frameRef ? d.frameRef.current : d.frame;

        if (currentFrame) {
          if (currentFrame !== previous) {
            instance.setFrame(currentFrame);
            previous = currentFrame;
          }
        }
        poll = requestAnimationFrame(loop);
      };
      loop();
    } catch {
      setError(
        "WebGL is unavailable. Enable hardware acceleration or use a WebGL-capable browser.",
      );
    }
    const reset = (e) => {
      if (
        e.key?.toLowerCase() === "r" &&
        !/INPUT|SELECT|TEXTAREA/.test(e.target?.tagName)
      )
        engine.current?.resetCamera();
    };
    window.addEventListener("keydown", reset);
    return () => {
      cancelAnimationFrame(poll);
      window.removeEventListener("keydown", reset);
      engine.current?.dispose();
      engine.current = null;
    };
  }, [playback]);

  useEffect(() => {
    if (engine.current) {
      engine.current.setOptions(options);
    }
  }, [options]);

  // Mode navigation by horizontal swipe (touch) or horizontal scroll (trackpad).
  // Orbit, pan and zoom keep working: a gesture only becomes a swipe once it is
  // clearly horizontal, and any camera movement it caused is rolled back.
  const modeRef = useRef(options.mode);
  modeRef.current = options.mode;
  const changeRef = useRef(onModeChange);
  changeRef.current = onModeChange;
  useEffect(() => {
    const node = mount.current;
    if (!node || !onModeChange) return;
    const step = (direction) => {
      const index = MODES.indexOf(modeRef.current);
      const next = MODES[index + direction];
      if (index < 0 || !next) return;
      changeRef.current(next);
    };
    const touch = {
      active: false,
      decided: false,
      x: 0,
      y: 0,
      at: 0,
      camera: null,
    };
    const onStart = (e) => {
      if (e.touches.length !== 1) {
        touch.active = false;
        return;
      }
      touch.active = true;
      touch.decided = false;
      touch.x = e.touches[0].clientX;
      touch.y = e.touches[0].clientY;
      touch.at = e.timeStamp;
      touch.camera = engine.current?.cameraState();
    };
    const onMove = (e) => {
      if (!touch.active || touch.decided || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - touch.x;
      const dy = e.touches[0].clientY - touch.y;
      if (Math.abs(dx) < 56 || Math.abs(dx) < Math.abs(dy) * 1.8) return;
      if (e.timeStamp - touch.at > 800) {
        touch.active = false;
        return;
      }
      touch.decided = true;
      engine.current?.restoreCamera(touch.camera);
      engine.current?.setControlsEnabled(false);
      step(dx < 0 ? 1 : -1);
    };
    const onEnd = () => {
      touch.active = false;
      engine.current?.setControlsEnabled(true);
    };
    let accumulated = 0,
      lastWheel = 0;
    const onWheel = (e) => {
      if (Math.abs(e.deltaX) <= Math.abs(e.deltaY) * 1.5) return;
      if (e.timeStamp - lastWheel > 500) accumulated = 0;
      lastWheel = e.timeStamp;
      accumulated += e.deltaX;
      if (Math.abs(accumulated) < 120) return;
      step(accumulated < 0 ? -1 : 1);
      accumulated = 0;
    };
    node.addEventListener("touchstart", onStart, { passive: true });
    node.addEventListener("touchmove", onMove, { passive: true });
    node.addEventListener("touchend", onEnd, { passive: true });
    node.addEventListener("touchcancel", onEnd, { passive: true });
    node.addEventListener("wheel", onWheel, { passive: true });
    return () => {
      node.removeEventListener("touchstart", onStart);
      node.removeEventListener("touchmove", onMove);
      node.removeEventListener("touchend", onEnd);
      node.removeEventListener("touchcancel", onEnd);
      node.removeEventListener("wheel", onWheel);
    };
  }, [onModeChange]);

  let visible = 0;
  if (summary?.heights) {
    const minH = options.minHeight;
    const maxH = options.maxHeight;
    const hArr = summary.heights;
    const numH = hArr.length;
    for (let i = 0; i < numH; i++) {
      if (hArr[i] >= minH && hArr[i] <= maxH) visible++;
    }
  }

  const visibleCellsCount = visible?.toLocaleString() ?? "—";
  const activeObjectsCount = summary?.objects ?? "—";

  return (
    <section
      className={`viewport premium-viewport ${playback ? "playback-viewport" : ""}`}
      aria-label="LiDAR visualization"
    >
      <div className="viewport-top">
        <div>
          <span className="eyebrow">{label}</span>
          <h2>{options.mode}</h2>
        </div>
        <div className="frame-info">
          <span>
            FRAME <b>{summary?.id ?? "—"}</b>
          </span>
          <span>
            {summary
              ? new Date(summary.timestamp).toLocaleTimeString("en-GB", {
                  hour12: false,
                })
              : "—"}
          </span>
        </div>
      </div>
      <div className="canvas-mount" ref={mount} />
      {(!summary || error) && (
        <div className="scene-message" role="status">
          <strong>
            {error
              ? "Renderer unavailable"
              : playback
                ? "Select a recording"
                : "Connecting to telemetry"}
          </strong>
          <span>
            {error ||
              (playback
                ? "Record or import a session to inspect it."
                : "Simulation starts after 3 seconds without a valid frame.")}
          </span>
        </div>
      )}
      {summary && !error && visible === 0 && (
        <div className="scene-message">
          <strong>No cells in this height range</strong>
          <span>
            Adjust the height filter or inspect the incoming cell array.
          </span>
        </div>
      )}
      <button
        className="camera-reset"
        onClick={() => engine.current?.resetCamera()}
        aria-label="Reset camera"
      >
        <Icon name="reset" />
        Reset view
      </button>
      <div className="scene-caption">
        <span>
          {options.mode === MODE_POINTS
            ? "PHYSICAL SENSOR POINT CLOUD · SPECTRAL ELEVATION GRADIENT"
            : options.mode === MODE_SEMANTIC
              ? "FLAT COST CLASSIFICATION · CLASSIFIED TRACKS"
              : options.mode === MODE_PERCEPTION
                ? "VEHICLE-CENTRIC PERCEPTION · ENVIRONMENT DERIVED FROM RECEIVED CELLS"
                : "INSTANCED 2.5D TERRAIN · DRDO DEFENSE PALETTE (TRI-COLOR)"}
        </span>
        <span>Hazards: 2 s velocity projection · not collision prediction</span>
      </div>
      <div className="viewport-bottom">
        <div>
          <span>{visibleCellsCount} visible cells</span>
          <em>·</em>
          <span>{activeObjectsCount} objects</span>
        </div>
        <div>
          <span>
            Drag: orbit <em>·</em> Right-drag: pan <em>·</em> Scroll: zoom
            {onModeChange ? (
              <>
                <em>·</em> Swipe: mode
              </>
            ) : (
              ""
            )}
          </span>
        </div>
      </div>
    </section>
  );
}
