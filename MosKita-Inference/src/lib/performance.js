export const METRIC_WINDOW = 90;

export function createEmptyPerformance() {
  return {
    currentFps: 0,
    averageFps: 0,
    lastLatencyMs: 0,
    averageLatencyMs: 0,
    p95LatencyMs: 0,
    framesProcessed: 0,
    lastDetectionCount: 0,
  };
}

export function pushWindowSample(samples, value, limit = METRIC_WINDOW) {
  if (!Number.isFinite(value) || value <= 0) {
    return samples;
  }

  const nextSamples = [...samples, value];
  return nextSamples.length > limit
    ? nextSamples.slice(nextSamples.length - limit)
    : nextSamples;
}

export function percentile(values, fraction) {
  if (!values.length) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);

  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }

  const weight = position - lowerIndex;
  return sorted[lowerIndex] + (sorted[upperIndex] - sorted[lowerIndex]) * weight;
}

function average(values) {
  if (!values.length) {
    return 0;
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
}

export function summarizePerformance({
  latencySamples = [],
  fpsSamples = [],
  framesProcessed = 0,
  lastDetectionCount = 0,
}) {
  return {
    currentFps: fpsSamples.at(-1) ?? 0,
    averageFps: average(fpsSamples),
    lastLatencyMs: latencySamples.at(-1) ?? 0,
    averageLatencyMs: average(latencySamples),
    p95LatencyMs: percentile(latencySamples, 0.95),
    framesProcessed,
    lastDetectionCount,
  };
}