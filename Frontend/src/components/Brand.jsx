import React, { useState, useEffect } from "react";

/**
 * DRISHTI-2.5D
 * Dynamic Real-Time Ingestion & Spatial Hazard Tracking Interface
 * Official Brand Identity Component
 */

export function DrishtiSymbol({ size = 40, className = "", style = {} }) {
  return (
    <div
      className={`drishti-symbol-wrapper ${className}`}
      style={{
        width: size,
        height: size,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        flexShrink: 0,
        ...style,
      }}
      aria-hidden="true"
    >
      <img
        src="/brand/drishti-symbol-transparent.png"
        alt="DRISHTI-2.5D Emblem"
        width={size}
        height={size}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          filter: "drop-shadow(0 2px 10px rgba(196, 172, 118, 0.22))",
          userSelect: "none",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}

// Backward compatibility alias for existing components
export const Logo = DrishtiSymbol;

/**
 * Spatial Geometric Object for Hero Right Side
 * Clean, subtle floating movement, gentle depth, ambient lighting,
 * and high-technology restraint without redundant typography.
 */
export function DrishtiSpatialSymbol({ size = 190 }) {
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width - 0.5) * 8;
    const y = ((e.clientY - rect.top) / rect.height - 0.5) * -8;
    setTilt({ x, y });
  };

  const handleMouseLeave = () => {
    setTilt({ x: 0, y: 0 });
  };

  return (
    <div
      className="drishti-spatial-container"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        perspective: "1000px",
      }}
    >
      <div
        className="drishti-hero-floating"
        style={{
          transform: `rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)`,
          transition: "transform 0.25s cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      >
        <div className="drishti-ambient-glow" />
        <div className="drishti-symbol-stage" style={{ width: size, height: size }}>
          <img
            src="/brand/drishti-symbol-transparent.png"
            alt="DRISHTI-2.5D Spatial Element"
            className="drishti-symbol-img"
            width={size}
            height={size}
          />
        </div>
      </div>
    </div>
  );
}

export const DrishtiHero = DrishtiSpatialSymbol;

export function Icon({ name, size = 19, ...props }) {
  const paths = {
    live: (
      <>
        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    analysis: (
      <>
        <path d="M4 19h16M6 15V9m6 6V4m6 11v-6" />
      </>
    ),
    playback: <path d="m8 4 12 8-12 8Z" />,
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="m9 3 1-1h4l1 4 3 1 3-1 2 4-3 3v3l1 3-4 2-3-2h-3l-3 1-2-4 2-3v-3L5 7l4-2Z" />
      </>
    ),
    arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
    back: <path d="M20 12H5m6-6-6 6 6 6" />,
    record: <circle cx="12" cy="12" r="6" />,
    stop: <rect x="5" y="5" width="14" height="14" rx="2" />,
    pause: (
      <>
        <path d="M8 4v16M16 4v16" />
      </>
    ),
    step: (
      <>
        <path d="m5 4 10 8-10 8Z" fill="currentColor" />
        <path d="M19 5v14" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5" />
      </>
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    layers: (
      <>
        <path d="m12 3 10 6-10 6L2 9Zm-10 10 10 6 10-6M2 17l10 6 10-6" />
      </>
    ),
    check: <path d="m5 12 4 4L20 5" />,
    reset: (
      <>
        <path d="M3 10a9 9 0 1 1 1 8M3 3v7h7" />
      </>
    ),
    shield: (
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name] || paths.live}
    </svg>
  );
}
