import {
  createEmptyPerformance,
  percentile,
  pushWindowSample,
  summarizePerformance,
} from './performance';

describe('performance helpers', () => {
  test('creates an empty performance snapshot', () => {
    expect(createEmptyPerformance()).toEqual({
      currentFps: 0,
      averageFps: 0,
      lastLatencyMs: 0,
      averageLatencyMs: 0,
      p95LatencyMs: 0,
      framesProcessed: 0,
      lastDetectionCount: 0,
    });
  });

  test('keeps only the latest samples inside the rolling window', () => {
    const samples = [12, 14, 16];
    expect(pushWindowSample(samples, 18, 3)).toEqual([14, 16, 18]);
  });

  test('calculates percentile with interpolation', () => {
    expect(percentile([10, 20, 30, 40], 0.75)).toBe(32.5);
  });

  test('summarizes latency and fps statistics', () => {
    expect(
      summarizePerformance({
        latencySamples: [52, 48, 50],
        fpsSamples: [12, 11, 10],
        framesProcessed: 3,
        lastDetectionCount: 2,
      }),
    ).toEqual({
      currentFps: 10,
      averageFps: 11,
      lastLatencyMs: 50,
      averageLatencyMs: 50,
      p95LatencyMs: 51.8,
      framesProcessed: 3,
      lastDetectionCount: 2,
    });
  });
});