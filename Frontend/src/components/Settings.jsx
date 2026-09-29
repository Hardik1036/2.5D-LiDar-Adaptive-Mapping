import React, { useEffect, useRef } from "react";
import { Icon } from "./Brand.jsx";
import { DEFAULT_OPTIONS } from "../three/Scene.js";
export default function Settings({ onClose, options, setOptions }) {
  const dialog = useRef(null);
  useEffect(() => {
    dialog.current.showModal();
    return () => dialog.current?.close();
  }, []);
  return (
    <dialog ref={dialog} className="settings-dialog" onCancel={onClose}>
      <div className="card-heading-row">
        <div>
          <span className="overline">LOCAL PREFERENCES</span>
          <h2>Workspace settings</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close settings"
        >
          <Icon name="close" />
        </button>
      </div>
      <p className="hint">These controls affect this browser session only.</p>
      <label className="settings-select">
        Render pixel ratio
        <select
          value={options.pixelRatio || 1.5}
          onChange={(e) =>
            setOptions((o) => ({ ...o, pixelRatio: +e.target.value }))
          }
        >
          <option value="1">1× · lower GPU load</option>
          <option value="1.5">1.5× · balanced</option>
          <option value="2">2× · sharper display</option>
        </select>
      </label>
      <label className="check-row">
        <span>Animate humanoid gait</span>
        <input
          type="checkbox"
          checked={options.motion !== false}
          onChange={(e) =>
            setOptions((o) => ({ ...o, motion: e.target.checked }))
          }
        />
      </label>
      <p className="hint">
        OS reduced-motion settings take priority. Telemetry positions always
        continue updating.
      </p>
      <div className="settings-note">
        <h3>Data connection</h3>
        <p>Local: ws://127.0.0.1:8765</p>
        <p>Hosted: Railway WebSocket</p>
        <span>
          Connection details and schema normalization live in
          telemetryService.js. No backend settings are changed here.
        </span>
      </div>
      <button
        className="secondary-button"
        onClick={() =>
          setOptions({ ...DEFAULT_OPTIONS, pixelRatio: 1.5, motion: true })
        }
      >
        Reset scene preferences
      </button>
    </dialog>
  );
}
