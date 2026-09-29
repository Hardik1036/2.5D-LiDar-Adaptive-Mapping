import React, { useState, useEffect } from "react";
import { DrishtiSymbol } from "./Brand.jsx";

export default function SystemLoader({ onComplete, duration = 1200 }) {
  const [fade, setFade] = useState(false);

  useEffect(() => {
    const fadeTimer = setTimeout(() => {
      setFade(true);
    }, duration - 350);

    const completeTimer = setTimeout(() => {
      onComplete?.();
    }, duration);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(completeTimer);
    };
  }, [duration, onComplete]);

  return (
    <div
      className={`system-loader-overlay ${fade ? "loader-fade-out" : ""}`}
      role="alert"
      aria-busy="true"
      aria-label="Initializing DRISHTI-2.5D"
    >
      <div className="system-loader-content">
        <div className="loader-symbol-pulse">
          <DrishtiSymbol size={96} />
        </div>
        <div className="loader-meta">
          <strong className="loader-title">DRISHTI-2.5D</strong>
          <span className="loader-subtitle">
            INITIALIZING SPATIAL HAZARD TRACKING INTERFACE
          </span>
          <div className="loader-bar">
            <span className="loader-progress" />
          </div>
        </div>
      </div>
    </div>
  );
}
