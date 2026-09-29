import React, { useEffect, useState } from "react";
import { DrishtiSymbol, Icon } from "./Brand.jsx";
export default function Header({
  status,
  connectionState,
  page,
  navigate,
  onSettings,
  isManuallyDisconnected,
  onToggleBackend,
}) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const isConnected = !isManuallyDisconnected && (status === "live" || status === "connected");

  const stateClass = isManuallyDisconnected
    ? "standby"
    : isConnected
      ? "connected"
      : status === "connecting"
        ? "connecting"
        : status === "simulated"
          ? "simulated"
          : "disconnected";

  const stateLabel =
    page === "playback"
      ? `FEED: ${status.toUpperCase()}`
      : isManuallyDisconnected
        ? "○ STANDBY (OFFLINE)"
        : isConnected
          ? "● CONNECTED"
          : status === "connecting"
            ? "CONNECTING"
            : connectionState || "DISCONNECTED";

  return (
    <header className="premium-header">
      <button
        className="brand-home"
        onClick={() => navigate("welcome")}
        aria-label="DRISHTI-2.5D Home"
      >
        <DrishtiSymbol size={32} />
        <span>
          <strong>DRISHTI-2.5D</strong>
          <small>DYNAMIC REAL-TIME INGESTION &amp; SPATIAL HAZARD TRACKING INTERFACE</small>
        </span>
      </button>
      <nav className="main-navigation" aria-label="Workspace navigation">
        {[
          ["live", "Live Visualization"],
          ["analysis", "Analysis"],
          ["playback", "Playback"],
        ].map(([key, name]) => (
          <button
            key={key}
            aria-current={page === key ? "page" : undefined}
            onClick={() => navigate(key)}
          >
            <Icon name={key} />
            <span>{name}</span>
          </button>
        ))}
        <button onClick={onSettings}>
          <Icon name="settings" />
          <span>Settings</span>
        </button>
      </nav>
      <div className="header-meta">
        {page === "welcome" ? (
          <span className="briefing-label">
            DRDO <span>|</span> IDEX
          </span>
        ) : (
          <div className="status-container">
            <div
              className={`status ${stateClass}`}
              role="status"
              aria-live="polite"
              title={
                isManuallyDisconnected
                  ? "Backend manually disconnected (Procedural simulation active)"
                  : status === "live"
                    ? "Connected to Python LiDAR perception backend (ws://127.0.0.1:8765)"
                    : status === "simulated"
                      ? "Simulation fallback active (Backend disconnected)"
                      : status === "connecting"
                        ? "Connecting to backend ws://127.0.0.1:8765..."
                        : "Disconnected from backend. Retrying with exponential backoff..."
              }
            >
              {!isManuallyDisconnected && !isConnected && <span />}
              {stateLabel}
              {(status === "simulated" || isManuallyDisconnected) && page !== "playback" && (
                <span className="sim-badge" aria-hidden="true">
                  SIM
                </span>
              )}
            </div>
            {page !== "playback" && onToggleBackend && (
              <button
                type="button"
                className={`backend-toggle-btn ${isManuallyDisconnected ? "is-reconnect" : "is-disconnect"}`}
                onClick={onToggleBackend}
                title={
                  isManuallyDisconnected
                    ? "Reconnect to Python LiDAR backend (ws://127.0.0.1:8765)"
                    : "Disconnect from live backend and switch to procedural simulation"
                }
              >
                <Icon name={isManuallyDisconnected ? "reset" : "close"} size={12} />
                <span>{isManuallyDisconnected ? "CONNECT" : "DISCONNECT"}</span>
              </button>
            )}
          </div>
        )}
        <div className="clock">
          <time dateTime={now.toISOString()}>
            {now.toLocaleTimeString("en-GB", { hour12: false })}
          </time>
          <span>
            {now.toLocaleDateString("en-GB", {
              day: "2-digit",
              month: "short",
              year: "numeric",
            })}
          </span>
        </div>
      </div>
    </header>
  );
}
