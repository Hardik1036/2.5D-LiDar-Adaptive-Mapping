import React from "react";

const RADIUS = 28;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

/**
 * AccuracyMeter - Tactical Radial Perception Fidelity Gauge
 * DRISHTI-2.5D Cockpit HUD
 *
 * @param {Object} props
 * @param {number|null} props.value Numerical percentage (0.0 to 100.0) or null if unlinked
 * @param {string} props.label Identifier label (default: "PERCEPTION FIDELITY")
 */
export default function AccuracyMeter({
  value = null,
  label = "POINTPILLARS mAP",
  status: statusProp = null,
  subtitle = "NuScenes validation · 81 samples",
}) {
  const isUnlinked = value === null || value === undefined || isNaN(Number(value));
  const score = isUnlinked ? null : Math.min(100, Math.max(0, Number(value)));

  const status = statusProp || (isUnlinked ? "NO VALIDATION DATA" : "OFFLINE VALIDATION");

  let color = "#58a6ff";
  let badgeBg = "rgba(88, 166, 255, 0.12)";
  let badgeBorder = "rgba(88, 166, 255, 0.35)";
  let readout = "--.-%";
  let offset = CIRCUMFERENCE;

  if (isUnlinked) {
    color = "#475569";
    badgeBg = "rgba(71, 85, 105, 0.18)";
    badgeBorder = "rgba(71, 85, 105, 0.45)";
    readout = "--.-%";
    offset = CIRCUMFERENCE;
  } else {
    readout = `${score.toFixed(1)}%`;
    offset = CIRCUMFERENCE - (score / 100) * CIRCUMFERENCE;
    if (status === "OFFLINE VALIDATION") {
      color = "#58a6ff";
      badgeBg = "rgba(88, 166, 255, 0.12)";
      badgeBorder = "rgba(88, 166, 255, 0.35)";
    } else if (score < 75.0) {
      color = "#FF1744";
      badgeBg = "rgba(255, 23, 68, 0.12)";
      badgeBorder = "rgba(255, 23, 68, 0.3)";
    } else if (score < 90.0) {
      color = "#FFD600";
      badgeBg = "rgba(255, 214, 0, 0.12)";
      badgeBorder = "rgba(255, 214, 0, 0.3)";
    } else {
      color = "#00E676";
      badgeBg = "rgba(0, 230, 118, 0.12)";
      badgeBorder = "rgba(0, 230, 118, 0.3)";
    }
  }

  const statusTextColor = isUnlinked ? "#94A3B8" : color;

  return (
    <div
      className="accuracy-meter-card"
      style={{
        background: "rgba(10, 17, 40, 0.85)",
        border: "1px solid rgba(51, 65, 85, 0.6)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        borderRadius: "8px",
        padding: "12px 14px",
        display: "flex",
        alignItems: "center",
        gap: "14px",
        boxShadow: "0 4px 20px rgba(0, 0, 0, 0.35)",
      }}
    >
      <div
        style={{
          position: "relative",
          width: "72px",
          height: "72px",
          flexShrink: 0,
        }}
      >
        <svg
          viewBox="0 0 72 72"
          width="72"
          height="72"
          style={{ display: "block" }}
          aria-hidden="true"
        >
          {/* Base Inactive Track */}
          <circle
            cx="36"
            cy="36"
            r={RADIUS}
            fill="none"
            stroke={isUnlinked ? "#475569" : "#1E293B"}
            strokeWidth="5"
          />
          {/* Active Colored Arc */}
          <circle
            cx="36"
            cy="36"
            r={RADIUS}
            fill="none"
            stroke={color}
            strokeWidth="5"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 36 36)"
            style={{
              transition: "stroke-dashoffset 0.35s ease, stroke 0.35s ease",
            }}
          />
          {/* Centered Monospace Readout */}
          <text
            x="36"
            y="36"
            textAnchor="middle"
            dominantBaseline="central"
            fill={isUnlinked ? "#94A3B8" : "#FFFFFF"}
            fontSize={isUnlinked ? "11.5" : "12.5"}
            fontWeight="700"
            fontFamily="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
            letterSpacing="-0.5px"
          >
            {readout}
          </text>
        </svg>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "5px",
          minWidth: 0,
          flex: 1,
        }}
      >
        <span
          style={{
            fontSize: "11px",
            fontWeight: "600",
            letterSpacing: "1.2px",
            textTransform: "uppercase",
            color: "#94A3B8",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span
            style={{
              fontSize: "10.5px",
              fontWeight: "700",
              letterSpacing: "0.8px",
              color: statusTextColor,
              background: badgeBg,
              border: `1px solid ${badgeBorder}`,
              borderRadius: "4px",
              padding: "2px 7px",
              textTransform: "uppercase",
              display: "inline-flex",
              alignItems: "center",
              gap: "5px",
            }}
          >
            <span
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: statusTextColor,
                display: "inline-block",
                boxShadow: isUnlinked ? "none" : `0 0 6px ${color}`,
              }}
            />
            {status}
          </span>
          <small style={{ fontSize: "11px", color: "#64748B" }}>
            {subtitle}
          </small>
        </div>
      </div>
    </div>
  );
}
