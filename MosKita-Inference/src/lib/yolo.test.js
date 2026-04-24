import {
  applyNms,
  decodeYoloOutput,
  intersectionOverUnion,
  resolveOutputShape,
} from './yolo';

describe('YOLO decoding helpers', () => {
  test('infers output layout for channel-first tensors', () => {
    expect(resolveOutputShape([1, 6, 8400])).toEqual({
      rows: 8400,
      channels: 6,
      layout: 'channels-first',
    });
  });

  test('infers output layout for channel-last tensors', () => {
    expect(resolveOutputShape([1, 8400, 6])).toEqual({
      rows: 8400,
      channels: 6,
      layout: 'channels-last',
    });
  });

  test('suppresses overlapping boxes of the same class', () => {
    const detections = [
      { x: 10, y: 10, width: 50, height: 50, score: 0.91, classId: 0 },
      { x: 15, y: 14, width: 48, height: 48, score: 0.84, classId: 0 },
      { x: 180, y: 180, width: 20, height: 20, score: 0.77, classId: 1 },
    ];

    expect(applyNms(detections, 0.3)).toHaveLength(2);
  });

  test('computes intersection over union', () => {
    const left = { x: 0, y: 0, width: 50, height: 50 };
    const right = { x: 25, y: 25, width: 50, height: 50 };
    expect(intersectionOverUnion(left, right)).toBeCloseTo(0.142857, 5);
  });

  test('decodes and clips detections from a channel-first tensor', () => {
    const detections = decodeYoloOutput(
      {
        dims: [1, 6, 2],
        data: Float32Array.from([
          100, 102,
          100, 102,
          50, 48,
          40, 38,
          0.9, 0.88,
          0.1, 0.12,
        ]),
      },
      {
        classNames: ['bucket', 'drum'],
        confidenceThreshold: 0.4,
        iouThreshold: 0.3,
        letterbox: {
          scale: 1,
          padX: 0,
          padY: 0,
          originalWidth: 640,
          originalHeight: 480,
        },
      },
    );

    expect(detections).toHaveLength(1);
    expect(detections[0]).toMatchObject({
      classId: 0,
      className: 'bucket',
      x: 75,
      y: 80,
      width: 50,
      height: 40,
    });
    expect(detections[0].score).toBeCloseTo(0.9, 5);
  });

  test('decodes channel-last tensors and maps through letterboxing', () => {
    const detections = decodeYoloOutput(
      {
        dims: [1, 2, 6],
        data: Float32Array.from([
          40, 50, 30, 20, 0.2, 0.82,
          120, 90, 40, 30, 0.78, 0.1,
        ]),
      },
      {
        classNames: ['drain_inlet', 'bucket'],
        confidenceThreshold: 0.5,
        iouThreshold: 0.45,
        letterbox: {
          scale: 0.5,
          padX: 10,
          padY: 20,
          originalWidth: 200,
          originalHeight: 150,
        },
      },
    );

    expect(detections).toHaveLength(2);
    expect(detections[0]).toMatchObject({
      classId: 1,
      className: 'bucket',
      x: 30,
      y: 40,
      width: 60,
      height: 40,
    });
    expect(detections[0].score).toBeCloseTo(0.82, 5);
  });
});