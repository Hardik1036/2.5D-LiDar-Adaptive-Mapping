export const MAX_RECORDING_BYTES = 30 * 1024 * 1024;
export const MAX_RECORDING_FRAMES = 600;
export const MAX_HISTORY = 300;
export function summarizeFrame(f) {
  const bands = [0, 0, 0];
  const cells = f?.cells || [];
  const len = cells.length;
  const heights = new Float32Array(len);
  let safeCount = 0;
  for (let i = 0; i < len; i++) {
    const c = cells[i];
    const cost = c.cost ?? 0;
    bands[cost <= 50 ? 0 : cost <= 180 ? 1 : 2]++;
    if (cost < 50) safeCount++;
    heights[i] = c.z_min;
  }
  const rawAcc =
    f?.system_stats?.tracking_accuracy ?? f?.system_stats?.accuracy;
  let accuracy = null;
  if (rawAcc !== undefined && rawAcc !== null && !isNaN(rawAcc)) {
    accuracy = Math.min(100, Math.max(0, Number(rawAcc)));
  } else {
    // Zero fabrication: report null when ground truth is absent
    accuracy = null;
  }
  const stats = f?.system_stats
    ? {
        ...f.system_stats,
        accuracy,
        tracking_accuracy: accuracy,
      }
    : { accuracy, tracking_accuracy: accuracy };

  return {
    frame: f,
    stats,
    accuracy,
    id: f?.frame_id,
    timestamp: f?.timestamp,
    systemStatus: f?.system_status,
    bands,
    total: len,
    objects: f?.dynamic_objects ? f.dynamic_objects.length : 0,
    heights,
  };
}
export function downloadText(name, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function validateRecording(data, validateFrame) {
  if (
    !data ||
    data.version !== 1 ||
    !Array.isArray(data.frames) ||
    !data.frames.length ||
    data.frames.length > MAX_RECORDING_FRAMES
  )
    throw new Error("Use a version 1 recording with 1–600 frames.");
  let last = -1;
  for (const entry of data.frames) {
    if (
      !entry ||
      !Number.isFinite(entry.elapsed_ms) ||
      entry.elapsed_ms < last ||
      entry.elapsed_ms < 0 ||
      entry.elapsed_ms > 86400000 ||
      !["live", "simulated"].includes(entry.source) ||
      !validateFrame(entry.frame)
    )
      throw new Error(
        "Invalid frame, source label, or timeline in this recording.",
      );
    last = entry.elapsed_ms;
  }
  return data.frames;
}
