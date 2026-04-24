import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import { CLASS_COLORS, CLASS_NAMES } from './constants/classes';
import { createInputTensor } from './lib/input';
import {
  createEmptyPerformance,
  pushWindowSample,
  summarizePerformance,
} from './lib/performance';
import { decodeYoloOutput } from './lib/yolo';

const DEFAULT_MODEL_PATH = '/models/moskita.onnx';
const MODEL_INPUT_SIZE = 640;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = Math.min(4, Math.max(1, Math.floor((navigator.hardwareConcurrency ?? 2) / 2)));

function formatMetric(value, digits = 1, suffix = '') {
  if (!Number.isFinite(value) || value <= 0) {
    return '--';
  }

  return `${value.toFixed(digits)}${suffix}`;
}

function clearOverlay(canvas) {
  const context = canvas?.getContext('2d');
  if (!canvas || !context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawOverlay(video, canvas, detections) {
  if (!video || !canvas) {
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return;
  }

  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, width, height);
  context.lineJoin = 'round';
  context.textBaseline = 'top';
  context.font = `${Math.max(13, Math.round(width / 42))}px "IBM Plex Mono", monospace`;

  detections.forEach((detection) => {
    const color = CLASS_COLORS[detection.classId % CLASS_COLORS.length];
    const strokeWidth = Math.max(2, Math.round(width / 320));
    const label = `${detection.className} ${(detection.score * 100).toFixed(1)}%`;
    const labelWidth = context.measureText(label).width + 18;
    const labelHeight = 26;
    const labelX = detection.x;
    const labelY = Math.max(0, detection.y - labelHeight - 4);

    context.strokeStyle = color;
    context.lineWidth = strokeWidth;
    context.fillStyle = `${color}26`;
    context.strokeRect(detection.x, detection.y, detection.width, detection.height);
    context.fillRect(detection.x, detection.y, detection.width, detection.height);

    context.fillStyle = color;
    context.beginPath();
    context.roundRect(labelX, labelY, labelWidth, labelHeight, 8);
    context.fill();

    context.fillStyle = '#f7f4ed';
    context.fillText(label, labelX + 9, labelY + 5);
  });
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const inputCanvasRef = useRef(null);
  const sessionRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(0);
  const processingRef = useRef(false);
  const latencySamplesRef = useRef([]);
  const fpsSamplesRef = useRef([]);
  const lastCompletedAtRef = useRef(0);
  const framesProcessedRef = useRef(0);
  const confidenceRef = useRef(0.4);
  const iouRef = useRef(0.45);
  const uploadedVideoUrlRef = useRef('');

  const [mode, setMode] = useState('camera');
  const [cameraActive, setCameraActive] = useState(false);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState('');
  const [uploadedVideoName, setUploadedVideoName] = useState('');
  const [uploadedModelName, setUploadedModelName] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [modelState, setModelState] = useState({
    status: 'loading',
    label: 'Default public model',
    error: '',
  });
  const [sourceState, setSourceState] = useState({
    ready: false,
    label: 'Waiting for input',
    error: '',
  });
  const [performanceState, setPerformanceState] = useState(createEmptyPerformance());
  const [detections, setDetections] = useState([]);
  const [runtimeError, setRuntimeError] = useState('');

  confidenceRef.current = confidenceThreshold;
  iouRef.current = iouThreshold;

  function resetPerformance() {
    latencySamplesRef.current = [];
    fpsSamplesRef.current = [];
    lastCompletedAtRef.current = 0;
    framesProcessedRef.current = 0;
    setPerformanceState(createEmptyPerformance());
  }

  async function loadModel(modelSource, label) {
    setModelState({ status: 'loading', label, error: '' });
    setRuntimeError('');

    try {
      const session = await ort.InferenceSession.create(modelSource, {
        executionProviders: ['wasm'],
      });

      sessionRef.current = session;
      resetPerformance();
      setDetections([]);
      setModelState({
        status: 'ready',
        label: `${label} · wasm`,
        error: '',
      });
    } catch (error) {
      sessionRef.current = null;
      setModelState({
        status: 'error',
        label,
        error: error?.message ?? 'Failed to load the ONNX model.',
      });
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setSourceState({
        ready: false,
        label: 'Camera unsupported',
        error: 'This browser does not expose navigator.mediaDevices.getUserMedia().',
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }

      setCameraActive(true);
      setSourceState({ ready: true, label: 'Rear camera live', error: '' });
      setRuntimeError('');
    } catch (error) {
      setCameraActive(false);
      setSourceState({
        ready: false,
        label: 'Camera unavailable',
        error: error?.message ?? 'The camera could not be opened.',
      });
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video && video.srcObject) {
      video.pause();
      video.srcObject = null;
    }

    setCameraActive(false);
    if (mode === 'camera') {
      setSourceState({ ready: false, label: 'Camera stopped', error: '' });
    }

    clearOverlay(overlayRef.current);
  }

  function revokeUploadedVideoUrl() {
    if (uploadedVideoUrlRef.current) {
      URL.revokeObjectURL(uploadedVideoUrlRef.current);
      uploadedVideoUrlRef.current = '';
    }
  }

  function handleVideoSelected(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    revokeUploadedVideoUrl();
    const nextUrl = URL.createObjectURL(file);
    uploadedVideoUrlRef.current = nextUrl;
    setUploadedVideoUrl(nextUrl);
    setUploadedVideoName(file.name);
    setMode('upload');
    resetPerformance();
    setDetections([]);
    setSourceState({ ready: false, label: file.name, error: '' });
    event.target.value = '';
  }

  async function handleModelSelected(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const buffer = await file.arrayBuffer();
    setUploadedModelName(file.name);
    await loadModel(buffer, `Uploaded model: ${file.name}`);
    event.target.value = '';
  }

  function handleReplay() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    video.currentTime = 0;
    resetPerformance();
    video.play().catch(() => undefined);
  }

  useEffect(() => {
    void loadModel(DEFAULT_MODEL_PATH, 'Default public model');
  }, []);

  useEffect(() => {
    if (mode === 'camera') {
      void startCamera();
      return () => {
        stopCamera();
      };
    }

    stopCamera();
    if (!uploadedVideoUrl) {
      setSourceState({ ready: false, label: 'Upload a video to begin', error: '' });
      clearOverlay(overlayRef.current);
    }

    return undefined;
  }, [mode, uploadedVideoUrl]);

  useEffect(() => {
    let cancelled = false;

    async function frameLoop() {
      if (cancelled) {
        return;
      }

      rafRef.current = requestAnimationFrame(frameLoop);

      const session = sessionRef.current;
      const video = videoRef.current;
      if (!session || !video || processingRef.current) {
        return;
      }

      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.paused || video.ended) {
        return;
      }

      processingRef.current = true;
      const frameStartedAt = performance.now();

      try {
        const { tensor, letterbox } = createInputTensor(video, {
          inputSize: MODEL_INPUT_SIZE,
          canvas: inputCanvasRef.current,
        });
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const parsedDetections = decodeYoloOutput(results[outputName], {
          classNames: CLASS_NAMES,
          confidenceThreshold: confidenceRef.current,
          iouThreshold: iouRef.current,
          letterbox,
        });

        drawOverlay(video, overlayRef.current, parsedDetections);

        const frameCompletedAt = performance.now();
        const frameLatency = frameCompletedAt - frameStartedAt;
        const currentFps = lastCompletedAtRef.current
          ? 1000 / (frameCompletedAt - lastCompletedAtRef.current)
          : 0;

        lastCompletedAtRef.current = frameCompletedAt;
        framesProcessedRef.current += 1;
        latencySamplesRef.current = pushWindowSample(latencySamplesRef.current, frameLatency);
        fpsSamplesRef.current = pushWindowSample(fpsSamplesRef.current, currentFps);

        setDetections(parsedDetections);
        setPerformanceState(
          summarizePerformance({
            latencySamples: latencySamplesRef.current,
            fpsSamples: fpsSamplesRef.current,
            framesProcessed: framesProcessedRef.current,
            lastDetectionCount: parsedDetections.length,
          }),
        );
        setRuntimeError('');
      } catch (error) {
        setRuntimeError(error?.message ?? 'Frame inference failed.');
      } finally {
        processingRef.current = false;
      }
    }

    rafRef.current = requestAnimationFrame(frameLoop);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      revokeUploadedVideoUrl();
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const modelHelp = modelState.status === 'error'
    ? 'Drop moskita.onnx into deploy/webapp/public/models/ or upload the exported model manually below.'
    : 'The app first tries /models/moskita.onnx, then keeps any uploaded model in memory for the current session.';

  const hasVisualSource = mode === 'camera' ? cameraActive : Boolean(uploadedVideoUrl);

  return (
    <div className="shell">
      <header className="card hero">
        <div>
          <p className="eyebrow">MosKita Edge Dashboard</p>
          <h1>Responsive React inference for field camera and uploaded video.</h1>
          <p className="lede">
            Run your exported ONNX detector in the browser, switch between live rear-camera capture
            and uploaded footage, and track throughput with frame-rate plus latency metrics.
          </p>
        </div>

        <div className="status-strip">
          <span className={`pill pill-${modelState.status}`}>
            Model: {modelState.status}
          </span>
          <span className={`pill ${sourceState.ready ? 'pill-ready' : 'pill-idle'}`}>
            Source: {sourceState.label}
          </span>
          <span className="pill pill-neutral">Input size: {MODEL_INPUT_SIZE}</span>
        </div>
      </header>

      <main className="layout">
        <section className="card main-stage">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Input Modes</p>
              <h2>Camera and video upload</h2>
            </div>

            <div className="segmented" role="tablist" aria-label="Input mode">
              <button
                type="button"
                className={mode === 'camera' ? 'mode-button active' : 'mode-button'}
                onClick={() => setMode('camera')}
              >
                Camera
              </button>
              <button
                type="button"
                className={mode === 'upload' ? 'mode-button active' : 'mode-button'}
                onClick={() => setMode('upload')}
              >
                Video Upload
              </button>
            </div>
          </div>

          {hasVisualSource ? (
            <div className="viewer-stack">
              <video
                ref={videoRef}
                className="viewer-media"
                muted
                playsInline
                autoPlay={mode === 'camera'}
                controls={mode === 'upload'}
                src={mode === 'upload' ? uploadedVideoUrl : undefined}
                onLoadedMetadata={() => {
                  const video = videoRef.current;
                  if (video?.videoWidth && video?.videoHeight) {
                    setSourceState({
                      ready: true,
                      label:
                        mode === 'camera'
                          ? `Rear camera ${video.videoWidth}×${video.videoHeight}`
                          : `${uploadedVideoName} · ${video.videoWidth}×${video.videoHeight}`,
                      error: '',
                    });
                    if (mode === 'upload') {
                      video.play().catch(() => undefined);
                    }
                  }
                }}
                onPause={() => clearOverlay(overlayRef.current)}
                onEnded={() => clearOverlay(overlayRef.current)}
              />
              <canvas ref={overlayRef} className="viewer-overlay" />
            </div>
          ) : (
            <div className="placeholder">
              <strong>{mode === 'camera' ? 'Camera preview is idle.' : 'Upload a video to start inference.'}</strong>
              <span>
                {mode === 'camera'
                  ? 'Rear camera permission is requested automatically and optimized for mobile.'
                  : 'Choose an .mp4, .mov, or .webm clip. Detections render over the video frame.'}
              </span>
            </div>
          )}

          <div className="viewer-actions">
            {mode === 'camera' ? (
              <button type="button" className="action-button" onClick={cameraActive ? stopCamera : startCamera}>
                {cameraActive ? 'Stop Camera' : 'Start Camera'}
              </button>
            ) : (
              <label className="action-button upload-button">
                Upload Video
                <input type="file" accept="video/*" onChange={handleVideoSelected} />
              </label>
            )}

            {mode === 'upload' && uploadedVideoUrl ? (
              <button type="button" className="action-button secondary" onClick={handleReplay}>
                Replay Clip
              </button>
            ) : null}

            <span className="helper-copy">Rear camera is requested with `facingMode: environment` for mobile browsers.</span>
          </div>

          {sourceState.error ? <p className="error-text">{sourceState.error}</p> : null}
          {runtimeError ? <p className="error-text">{runtimeError}</p> : null}
        </section>

        <aside className="sidebar">
          <section className="card sidebar-card">
            <p className="eyebrow">Model</p>
            <h2>Inference setup</h2>
            <p className="support-text">{modelHelp}</p>

            <div className="stack">
              <div className="info-block">
                <span className="label">Model source</span>
                <strong>{uploadedModelName || modelState.label}</strong>
              </div>

              <label className="upload-panel">
                <span>Upload ONNX model</span>
                <small>Use the exported `moskita.onnx` from your training run.</small>
                <input type="file" accept=".onnx,application/octet-stream" onChange={handleModelSelected} />
              </label>

              {modelState.error ? <p className="error-text">{modelState.error}</p> : null}

              <div className="slider-group">
                <label htmlFor="confidence-threshold">
                  Confidence threshold
                  <strong>{confidenceThreshold.toFixed(2)}</strong>
                </label>
                <input
                  id="confidence-threshold"
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
                />
              </div>

              <div className="slider-group">
                <label htmlFor="iou-threshold">
                  NMS IoU threshold
                  <strong>{iouThreshold.toFixed(2)}</strong>
                </label>
                <input
                  id="iou-threshold"
                  type="range"
                  min="0.1"
                  max="0.8"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(event) => setIouThreshold(Number(event.target.value))}
                />
              </div>
            </div>
          </section>

          <section className="card sidebar-card">
            <p className="eyebrow">Performance</p>
            <h2>FPS and latency</h2>

            <div className="metric-grid">
              <article className="metric-card highlight">
                <span>Current FPS</span>
                <strong>{formatMetric(performanceState.currentFps)}</strong>
              </article>
              <article className="metric-card">
                <span>Average FPS</span>
                <strong>{formatMetric(performanceState.averageFps)}</strong>
              </article>
              <article className="metric-card">
                <span>Last latency</span>
                <strong>{formatMetric(performanceState.lastLatencyMs, 1, ' ms')}</strong>
              </article>
              <article className="metric-card">
                <span>Average latency</span>
                <strong>{formatMetric(performanceState.averageLatencyMs, 1, ' ms')}</strong>
              </article>
              <article className="metric-card">
                <span>p95 latency</span>
                <strong>{formatMetric(performanceState.p95LatencyMs, 1, ' ms')}</strong>
              </article>
              <article className="metric-card">
                <span>Frames processed</span>
                <strong>{performanceState.framesProcessed || '--'}</strong>
              </article>
            </div>
          </section>

          <section className="card sidebar-card">
            <p className="eyebrow">Detections</p>
            <h2>Latest frame</h2>

            <div className="detection-summary">
              <span>Objects in last inference</span>
              <strong>{performanceState.lastDetectionCount}</strong>
            </div>

            <ul className="detection-list">
              {detections.length ? (
                detections.slice(0, 8).map((detection, index) => (
                  <li key={`${detection.className}-${index}`}>
                    <span
                      className="swatch"
                      style={{ backgroundColor: CLASS_COLORS[detection.classId % CLASS_COLORS.length] }}
                    />
                    <div>
                      <strong>{detection.className}</strong>
                      <small>
                        {(detection.score * 100).toFixed(1)}% · {Math.round(detection.width)}×{Math.round(detection.height)} px
                      </small>
                    </div>
                  </li>
                ))
              ) : (
                <li className="empty-state">No detections yet. Start the source and wait for the next processed frame.</li>
              )}
            </ul>

            <div className="class-footnote">
              <span>Classes</span>
              <strong>{CLASS_NAMES.join(' · ')}</strong>
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}