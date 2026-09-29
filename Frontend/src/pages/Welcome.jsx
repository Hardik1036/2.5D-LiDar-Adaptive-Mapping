import React, { useEffect, useRef, useState } from "react";
import { Icon, DrishtiHero } from "../components/Brand.jsx";
import { mountWelcomeScene } from "../three/WelcomeScene.js";

const slides = [
  {
    tag: "THE MISSION",
    title: (
      <>
        Clarity beyond
        <br />
        the <em>point cloud.</em>
      </>
    ),
    text: "A focused spatial workspace for dynamic real-time LiDAR ingestion and spatial hazard tracking. Built to make complex 2.5D elevation models and hazard telemetry immediately clear, navigable, and actionable.",
    label: "Understand the mission",
    detail: "Dynamic Real-Time Ingestion & Spatial Hazard Tracking Interface",
  },
  {
    tag: "THE ARCHITECTURE",
    title: (
      <>
        Less clutter.
        <br />
        <em>More context.</em>
      </>
    ),
    text: "Your backend performs the heavy spatial processing. DRISHTI-2.5D turns adaptive cells and tracked hazard vectors into a responsive, intuitive interface without masking sensor provenance.",
    label: "Understand the system",
    detail: "One telemetry contract. Multi-scale 2.5D perception.",
  },
  {
    tag: "READY TO OPERATE",
    title: (
      <>
        Connect. Inspect.
        <br />
        <em>Replay.</em>
      </>
    ),
    text: "Start with live visualization, examine incoming telemetry in real-time, and record bounded sessions for diagnostic review. Explicit simulation fallbacks keep operational briefings moving.",
    label: "Enter the workspace",
    detail: "Live when connected. Resilient when offline.",
  },
];

export default function Welcome({ onLaunch }) {
  const [step, setStep] = useState(0),
    [motion, setMotion] = useState(true),
    [heroError, setHeroError] = useState(false);
  const mount = useRef(null);

  useEffect(() => {
    if (!motion) return;
    try {
      return mountWelcomeScene(mount.current);
    } catch {
      setHeroError(true);
    }
  }, [motion]);

  const s = slides[step];

  return (
    <main className="welcome-page">
      <div className="welcome-topline">
        <span className="overline">
          SPATIAL DEFENSE INTELLIGENCE <b>/ DRISHTI-2.5D</b>
        </span>
        <button className="text-button" onClick={onLaunch}>
          Skip to workspace <Icon name="arrow" />
        </button>
      </div>

      <section className="welcome-hero" aria-label="Project welcome">
        <div className="welcome-copy" key={step}>
          <div className="slide-kicker">
            <span className="mono">0{step + 1} / 03</span>
            <span>{s.tag}</span>
          </div>
          <h1>{s.title}</h1>
          <p className="welcome-description">{s.text}</p>
          {step === 0 ? (
            <div className="mission-tags">
              <span>DRDO | IDEX</span>
              <span>Defense &amp; Autonomous Logistics</span>
            </div>
          ) : step === 1 ? (
            <div className="architecture-flow">
              <span>LiDAR Stream</span>
              <Icon name="arrow" />
              <span>Adaptive Ingestion</span>
              <Icon name="arrow" />
              <span>DRISHTI-2.5D Engine</span>
            </div>
          ) : (
            <div className="ready-list">
              <span>
                <Icon name="check" /> Explicit source status
              </span>
              <span>
                <Icon name="check" /> Local session recording &amp; replay
              </span>
            </div>
          )}

          <div className="welcome-actions">
            <button
              className="primary-button"
              onClick={() => (step < 2 ? setStep(step + 1) : onLaunch())}
            >
              {step === 2 ? "Launch workspace" : "Explore the system"}
              <Icon name="arrow" />
            </button>
            {step < 2 && (
              <button className="secondary-button" onClick={onLaunch}>
                Open live view
              </button>
            )}
          </div>
          <p className="hero-footnote">
            Frontend visualization · Real-time spatial perception interface
          </p>
        </div>

        <div className={`welcome-art ${motion ? "" : "motion-paused"}`}>
          <div className="hero-webgl" ref={mount} />

          <div className="hero-spatial-symbol">
            <DrishtiHero size={190} />
          </div>

          <div className="floating-label float-a">
            <Icon name="layers" />
            <div>
              <strong>Adaptive terrain</strong>
              <span>Variable-resolution cells</span>
            </div>
          </div>
          <div className="floating-label float-b">
            <span className="tracking-mark" />
            <div>
              <strong>Hazard tracking</strong>
              <span>Vector · velocity · classification</span>
            </div>
          </div>
          <div className="art-caption">
            <span>SPATIAL ARCHITECTURE</span>
            <span>Illustrative geometry · not live telemetry</span>
          </div>
          <button
            className="motion-button"
            aria-pressed={!motion}
            onClick={() => setMotion((v) => !v)}
          >
            {motion ? "Pause" : "Enable"} ambient motion
          </button>
        </div>
      </section>

      <nav className="welcome-steps" aria-label="Welcome slides">
        {slides.map((slide, i) => (
          <button
            key={slide.tag}
            aria-current={step === i ? "step" : undefined}
            onClick={() => setStep(i)}
          >
            <span className="step-number">0{i + 1}</span>
            <div>
              <strong>{slide.label}</strong>
              <span>{slide.detail}</span>
            </div>
            <Icon name="arrow" />
          </button>
        ))}
      </nav>

      <div className="welcome-bottom">
        <span>Designed for situational understanding.</span>
        <span>
          Built for a Safer, Smarter Tomorrow{" "}
          <span className="gold-text">/ DRISHTI-2.5D</span>
        </span>
      </div>
    </main>
  );
}
