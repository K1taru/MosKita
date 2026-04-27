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

const DEFAULT_MODEL_CANDIDATE_PATHS = Object.freeze([
  'models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx',
  'models/exports/moskita.onnx',
  'models/moskita.onnx',
]);
const MODEL_URL_OVERRIDE = (import.meta.env.VITE_MODEL_URL ?? '').trim();
const MODEL_INPUT_SIZE = 640;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.5;
const MODEL_LOAD_TIMEOUT_MS = 25000;
const CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: { ideal: 'environment' },
    width: { ideal: 1280 },
    height: { ideal: 720 },
  },
};

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = Math.min(4, Math.max(1, Math.floor((navigator.hardwareConcurrency ?? 2) / 2)));

function formatMetric(value, digits = 1, suffix = '') {
  if (!Number.isFinite(value) || value <= 0) {
    return '--';
  }

  return `${value.toFixed(digits)}${suffix}`;
}

function formatConfidenceScore(score) {
  if (!Number.isFinite(score)) {
    return '0.0%';
  }

  const boundedScore = Math.min(1, Math.max(0, score));
  return `${(boundedScore * 100).toFixed(1)}%`;
}

function normalizeBaseUrl(baseUrl) {
  if (!baseUrl || baseUrl === '/') {
    return '/';
  }

  return baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
}

function toAbsoluteUrl(pathOrUrl) {
  if (typeof window === 'undefined') {
    return pathOrUrl;
  }

  try {
    return new URL(pathOrUrl, window.location.href).toString();
  } catch {
    return pathOrUrl;
  }
}

function resolveDefaultModelUrls() {
  const resolvedUrls = new Set();
  const baseUrl = normalizeBaseUrl(import.meta.env.BASE_URL ?? '/');

  if (MODEL_URL_OVERRIDE) {
    resolvedUrls.add(toAbsoluteUrl(MODEL_URL_OVERRIDE));
  }

  DEFAULT_MODEL_CANDIDATE_PATHS.forEach((relativePath) => {
    const normalizedRelativePath = relativePath.replace(/^\/+/, '');
    resolvedUrls.add(toAbsoluteUrl(`${baseUrl}${normalizedRelativePath}`));
    resolvedUrls.add(toAbsoluteUrl(`/${normalizedRelativePath}`));
  });

  return [...resolvedUrls];
}

function decodeProbeText(arrayBuffer, byteLimit = 256) {
  const probeLength = Math.min(arrayBuffer.byteLength, byteLimit);
  const probeBytes = new Uint8Array(arrayBuffer.slice(0, probeLength));
  return new TextDecoder('utf-8').decode(probeBytes).trim().toLowerCase();
}

function formatByteLength(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function validateModelPayload(arrayBuffer, contentType, sourceUrl) {
  if (!arrayBuffer || arrayBuffer.byteLength < 1024) {
    throw new Error(`Downloaded file is too small (${formatByteLength(arrayBuffer?.byteLength ?? 0)}).`);
  }

  const probeText = decodeProbeText(arrayBuffer);
  if (probeText.startsWith('version https://git-lfs.github.com/spec/v1')) {
    throw new Error('Downloaded file is a Git LFS pointer, not the actual ONNX binary.');
  }

  const looksLikeHtml = probeText.includes('<!doctype html')
    || probeText.includes('<html')
    || probeText.includes('<head')
    || probeText.includes('<body');
  const isHtmlContentType = (contentType ?? '').toLowerCase().includes('text/html');

  if (looksLikeHtml || isHtmlContentType) {
    throw new Error(`URL returned HTML instead of ONNX bytes (${sourceUrl}).`);
  }
}

function getModelLoadErrorMessage(error) {
  const rawMessage = error?.message ?? 'Failed to load the ONNX model.';
  const normalized = rawMessage.toLowerCase();

  if (normalized.includes('protobuf parsing failed')) {
    return 'Failed to parse ONNX bytes. This usually means the model URL returned HTML/404 content, a Git LFS pointer file, or a truncated upload.';
  }

  return rawMessage;
}

async function fetchModelArrayBuffer(modelUrl) {
  const response = await fetch(modelUrl, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching model`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const contentType = response.headers.get('content-type') ?? '';
  validateModelPayload(arrayBuffer, contentType, modelUrl);

  return {
    arrayBuffer,
    byteLength: arrayBuffer.byteLength,
  };
}

async function createSessionWithTimeout(modelSource) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`Model initialization timed out after ${Math.round(MODEL_LOAD_TIMEOUT_MS / 1000)}s.`));
    }, MODEL_LOAD_TIMEOUT_MS);
  });

  try {
    return await Promise.race([
      ort.InferenceSession.create(modelSource, {
        executionProviders: ['wasm'],
      }),
      timeoutPromise,
    ]);
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

function clearOverlay(canvas) {
  const context = canvas?.getContext('2d');
  if (!canvas || !context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
}

function getSourceDimensions(source) {
  return {
    width: source?.videoWidth || source?.naturalWidth || source?.width || 0,
    height: source?.videoHeight || source?.naturalHeight || source?.height || 0,
  };
}

function drawOverlay(source, canvas, detections) {
  if (!source || !canvas) {
    return;
  }

  const { width, height } = getSourceDimensions(source);
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
    const label = `${detection.className} ${formatConfidenceScore(detection.score)}`;
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

function getLegacyGetUserMedia() {
  if (typeof navigator === 'undefined') {
    return null;
  }

  return navigator.getUserMedia
    || navigator.webkitGetUserMedia
    || navigator.mozGetUserMedia
    || navigator.msGetUserMedia
    || null;
}

function getUnsupportedCameraMessage() {
  if (typeof window !== 'undefined' && window.isSecureContext === false) {
    return 'Camera access requires a secure context. Use HTTPS, or open the app via http://localhost.';
  }

  return 'This browser does not support camera capture APIs. Try a recent Chrome, Edge, Firefox, or Safari release.';
}

function getCameraOpenErrorMessage(error) {
  const name = error?.name;

  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Allow camera access in browser settings and try again.';
  }

  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No camera device was found on this system.';
  }

  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The camera is already in use by another application.';
  }

  if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
    return 'Requested camera settings are not supported on this device.';
  }

  if (name === 'SecurityError') {
    return 'Camera access requires HTTPS or localhost.';
  }

  return error?.message ?? 'The camera could not be opened.';
}

async function requestCameraStream(deviceId = '') {
  if (typeof navigator === 'undefined') {
    throw new Error('Camera APIs are unavailable in this environment.');
  }

  if (navigator.mediaDevices?.getUserMedia) {
    const videoConstraints = deviceId
      ? {
        ...CAMERA_CONSTRAINTS.video,
        deviceId: { exact: deviceId },
      }
      : CAMERA_CONSTRAINTS.video;

    return navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints,
    });
  }

  const legacyGetUserMedia = getLegacyGetUserMedia();
  if (legacyGetUserMedia) {
    return new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, CAMERA_CONSTRAINTS, resolve, reject);
    });
  }

  throw new Error(getUnsupportedCameraMessage());
}

export default function App() {
  const videoRef = useRef(null);
  const imageRef = useRef(null);
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
  const confidenceRef = useRef(DEFAULT_CONFIDENCE_THRESHOLD);
  const iouRef = useRef(0.45);
  const uploadedVideoUrlRef = useRef('');
  const uploadedImageUrlRef = useRef('');
  const uploadedModelBufferRef = useRef(null);

  const [mode, setMode] = useState('camera');
  const [cameraActive, setCameraActive] = useState(false);
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [refreshingDevices, setRefreshingDevices] = useState(false);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState('');
  const [uploadedVideoName, setUploadedVideoName] = useState('');
  const [uploadedImageUrl, setUploadedImageUrl] = useState('');
  const [uploadedImageName, setUploadedImageName] = useState('');
  const [uploadedModelName, setUploadedModelName] = useState('');
  const [modelVersion, setModelVersion] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
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

  async function refreshVideoDevices(preferredDeviceId = '') {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setVideoDevices([]);
      setSelectedDeviceId('');
      return [];
    }

    setRefreshingDevices(true);
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === 'videoinput');

      setVideoDevices(cameras);
      setSelectedDeviceId((currentDeviceId) => {
        if (preferredDeviceId && cameras.some((device) => device.deviceId === preferredDeviceId)) {
          return preferredDeviceId;
        }

        if (currentDeviceId && cameras.some((device) => device.deviceId === currentDeviceId)) {
          return currentDeviceId;
        }

        return cameras[0]?.deviceId ?? '';
      });

      return cameras;
    } finally {
      setRefreshingDevices(false);
    }
  }

  async function loadModel(modelSource, label) {
    setModelState({ status: 'loading', label, error: '' });
    setRuntimeError('');

    try {
      const session = await createSessionWithTimeout(modelSource);

      sessionRef.current = session;
      resetPerformance();
      setDetections([]);
      setModelVersion((version) => version + 1);
      setModelState({
        status: 'ready',
        label: `${label} · wasm`,
        error: '',
      });
      return { ok: true };
    } catch (error) {
      sessionRef.current = null;
      setModelState({
        status: 'error',
        label,
        error: getModelLoadErrorMessage(error),
      });
      return { ok: false, error };
    }
  }

  async function loadDefaultModel() {
    const attempts = [];
    const candidateUrls = resolveDefaultModelUrls();

    setModelState({ status: 'loading', label: 'Default model auto-detect', error: '' });
    setRuntimeError('');

    for (const modelUrl of candidateUrls) {
      try {
        const { arrayBuffer, byteLength } = await fetchModelArrayBuffer(modelUrl);
        const fileName = modelUrl.split('/').pop() || 'model.onnx';
        const loadResult = await loadModel(arrayBuffer, `Default model: ${fileName} (${formatByteLength(byteLength)})`);
        if (loadResult.ok) {
          return;
        }

        attempts.push(`${modelUrl} -> ${getModelLoadErrorMessage(loadResult.error)}`);
      } catch (error) {
        attempts.push(`${modelUrl} -> ${getModelLoadErrorMessage(error)}`);
      }
    }

    sessionRef.current = null;
    const conciseAttempts = attempts.slice(0, 3).join(' | ');
    setModelState({
      status: 'error',
      label: 'Default model auto-detect',
      error: `No valid ONNX model found at default paths. ${conciseAttempts} Copy your model to /models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx (or BASE_URL/models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx), then reload, or upload manually.`,
    });
  }

  async function refreshModelSession() {
    sessionRef.current = null;
    setDetections([]);
    resetPerformance();

    if (uploadedModelBufferRef.current) {
      await loadModel(
        uploadedModelBufferRef.current,
        `Uploaded model: ${uploadedModelName || 'custom_model.onnx'}`,
      );
      return;
    }

    await loadDefaultModel();
  }

  async function startCamera(preferredDeviceId = selectedDeviceId) {
    if (!navigator.mediaDevices?.getUserMedia && !getLegacyGetUserMedia()) {
      setSourceState({
        ready: false,
        label: 'Camera unsupported',
        error: getUnsupportedCameraMessage(),
      });
      return;
    }

    try {
      stopCamera();

      let stream;
      try {
        stream = await requestCameraStream(preferredDeviceId);
      } catch (error) {
        const shouldFallback = Boolean(preferredDeviceId)
          && (
            error?.name === 'OverconstrainedError'
            || error?.name === 'ConstraintNotSatisfiedError'
            || error?.name === 'NotFoundError'
            || error?.name === 'DevicesNotFoundError'
          );

        if (!shouldFallback) {
          throw error;
        }

        stream = await requestCameraStream();
      }

      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }

      const activeTrack = stream.getVideoTracks()[0];
      const activeDeviceId = activeTrack?.getSettings?.().deviceId ?? preferredDeviceId;
      const devices = await refreshVideoDevices(activeDeviceId);
      const activeDeviceLabel = devices.find((device) => device.deviceId === activeDeviceId)?.label;
      const sourceLabel = activeDeviceLabel || (preferredDeviceId ? 'Selected camera live' : 'Rear camera live');

      setCameraActive(true);
      setSourceState({ ready: true, label: sourceLabel, error: '' });
      setRuntimeError('');
    } catch (error) {
      setCameraActive(false);
      setSourceState({
        ready: false,
        label: 'Camera unavailable',
        error: getCameraOpenErrorMessage(error),
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

  function revokeUploadedImageUrl() {
    if (uploadedImageUrlRef.current) {
      URL.revokeObjectURL(uploadedImageUrlRef.current);
      uploadedImageUrlRef.current = '';
    }
  }

  async function runInferenceOnSource(source) {
    const session = sessionRef.current;
    if (!session || !source) {
      return;
    }

    const frameStartedAt = performance.now();
    const { tensor, letterbox } = createInputTensor(source, {
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

    drawOverlay(source, overlayRef.current, parsedDetections);

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
  }

  function handleVideoSelected(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    revokeUploadedVideoUrl();
    revokeUploadedImageUrl();
    const nextUrl = URL.createObjectURL(file);
    uploadedVideoUrlRef.current = nextUrl;
    setUploadedVideoUrl(nextUrl);
    setUploadedVideoName(file.name);
    setUploadedImageUrl('');
    setUploadedImageName('');
    setMode('video');
    resetPerformance();
    setDetections([]);
    setSourceState({ ready: false, label: file.name, error: '' });
    event.target.value = '';
  }

  function handleImageSelected(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    revokeUploadedVideoUrl();
    revokeUploadedImageUrl();
    const nextUrl = URL.createObjectURL(file);
    uploadedImageUrlRef.current = nextUrl;
    setUploadedImageUrl(nextUrl);
    setUploadedImageName(file.name);
    setUploadedVideoUrl('');
    setUploadedVideoName('');
    setMode('image');
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
    uploadedModelBufferRef.current = buffer;
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
    void loadDefaultModel();
  }, []);

  useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return undefined;
    }

    void refreshVideoDevices();
    const handleDeviceChange = () => {
      void refreshVideoDevices();
    };

    navigator.mediaDevices.addEventListener?.('devicechange', handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener?.('devicechange', handleDeviceChange);
    };
  }, []);

  useEffect(() => {
    if (mode === 'camera') {
      return () => {
        stopCamera();
      };
    }

    stopCamera();
    if (mode === 'video' && !uploadedVideoUrl) {
      setSourceState({ ready: false, label: 'Upload a video to begin', error: '' });
      clearOverlay(overlayRef.current);
    }
    if (mode === 'image' && !uploadedImageUrl) {
      setSourceState({ ready: false, label: 'Upload an image to begin', error: '' });
      clearOverlay(overlayRef.current);
    }

    return undefined;
  }, [mode, uploadedVideoUrl, uploadedImageUrl]);

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
      try {
        await runInferenceOnSource(video);
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
      revokeUploadedImageUrl();
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    const image = imageRef.current;
    if (mode !== 'image' || !uploadedImageUrl || !image?.complete) {
      return;
    }

    runInferenceOnSource(image).catch((error) => {
      setRuntimeError(error?.message ?? 'Image inference failed.');
    });
  }, [mode, uploadedImageUrl, modelVersion, confidenceThreshold, iouThreshold]);

  const modelHelp = modelState.status === 'error'
    ? 'Place a real ONNX binary at /models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx (or BASE_URL/models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx for static hosting), ensure it is not a Git LFS pointer file, then reload. Upload is also supported below.'
    : 'The app auto-tries default model paths (starting with /models/exports/moskita_moskita-v12_yolo26n_img640_ep70.onnx, then /models/exports/moskita.onnx, plus BASE_URL variants), then keeps uploaded ONNX models in memory for the current session.';

  const activeModelDisplayName = uploadedModelName || modelState.label;

  const hasVisualSource = mode === 'camera' || (mode === 'video' && uploadedVideoUrl) || (mode === 'image' && uploadedImageUrl);

  return (
    <div className="shell">
      <header className="card hero">
        <div>
          <p className="eyebrow">MosKita Edge Dashboard</p>
          <h1>Inference for Mosquito Breeding Site Detection</h1>
          <p className="lede">
            Run your exported ONNX detector in the browser, switch between live rear-camera capture
            uploaded footage, and still images while tracking throughput with frame-rate plus latency metrics.
          </p>
        </div>

        <div className="status-strip">
          <span className={`pill pill-${modelState.status}`}>
            Model: {modelState.status} · {activeModelDisplayName}
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
                className={mode === 'video' ? 'mode-button active' : 'mode-button'}
                onClick={() => setMode('video')}
              >
                Video Upload
              </button>
              <button
                type="button"
                className={mode === 'image' ? 'mode-button active' : 'mode-button'}
                onClick={() => setMode('image')}
              >
                Image Upload
              </button>
            </div>
          </div>

          {hasVisualSource ? (
            <div className="viewer-stack">
              {mode === 'image' ? (
                <img
                  ref={imageRef}
                  className="viewer-media"
                  src={uploadedImageUrl}
                  alt={uploadedImageName}
                  onLoad={(event) => {
                    const image = event.currentTarget;
                    setSourceState({
                      ready: true,
                      label: `${uploadedImageName} · ${image.naturalWidth}×${image.naturalHeight}`,
                      error: '',
                    });
                    runInferenceOnSource(image).catch((error) => {
                      setRuntimeError(error?.message ?? 'Image inference failed.');
                    });
                  }}
                />
              ) : (
                <video
                  ref={videoRef}
                  className="viewer-media"
                  muted
                  playsInline
                  autoPlay={mode === 'camera'}
                  controls={mode === 'video'}
                  src={mode === 'video' ? uploadedVideoUrl : undefined}
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
                      if (mode === 'video') {
                        video.play().catch(() => undefined);
                      }
                    }
                  }}
                  onPause={() => clearOverlay(overlayRef.current)}
                  onEnded={() => clearOverlay(overlayRef.current)}
                />
              )}
              <canvas ref={overlayRef} className="viewer-overlay" />
            </div>
          ) : (
            <div className="placeholder">
              <strong>{mode === 'image' ? 'Upload an image to start inference.' : 'Upload a video to start inference.'}</strong>
              <span>
                {mode === 'image'
                  ? 'Choose a .jpg, .png, or .webp image. Detections render over the still frame.'
                  : 'Choose an .mp4, .mov, or .webm clip. Detections render over the video frame.'}
              </span>
            </div>
          )}

          <div className="viewer-actions">
            {mode === 'camera' ? (
              <>
                <button type="button" className="action-button" onClick={cameraActive ? stopCamera : startCamera}>
                  {cameraActive ? 'Stop Camera' : 'Start Camera'}
                </button>

                <label className="device-select" htmlFor="camera-device">
                  Camera device
                  <select
                    id="camera-device"
                    value={selectedDeviceId}
                    onChange={(event) => {
                      const nextDeviceId = event.target.value;
                      setSelectedDeviceId(nextDeviceId);

                      if (cameraActive) {
                        void startCamera(nextDeviceId);
                      }
                    }}
                  >
                    {videoDevices.length ? (
                      videoDevices.map((device, index) => (
                        <option key={device.deviceId || `${device.kind}-${index}`} value={device.deviceId}>
                          {device.label || `Camera ${index + 1}`}
                        </option>
                      ))
                    ) : (
                      <option value="">{refreshingDevices ? 'Detecting cameras...' : 'Default camera'}</option>
                    )}
                  </select>
                </label>

                <button
                  type="button"
                  className="action-button secondary"
                  onClick={() => void refreshVideoDevices()}
                  disabled={refreshingDevices}
                >
                  {refreshingDevices ? 'Refreshing Cameras...' : 'Refresh Cameras'}
                </button>
              </>
            ) : mode === 'video' ? (
              <label className="action-button upload-button">
                Upload Video
                <input type="file" accept="video/*" onChange={handleVideoSelected} />
              </label>
            ) : (
              <label className="action-button upload-button">
                Upload Image
                <input type="file" accept="image/*" onChange={handleImageSelected} />
              </label>
            )}

            {mode === 'video' && uploadedVideoUrl ? (
              <button type="button" className="action-button secondary" onClick={handleReplay}>
                Replay Clip
              </button>
            ) : null}

            <span className="helper-copy">Select any detected camera device, then start capture. Camera APIs need HTTPS or localhost.</span>
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

              <button type="button" className="action-button secondary" onClick={() => void refreshModelSession()}>
                Refresh Model Session
              </button>

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
                        {formatConfidenceScore(detection.score)} · {Math.round(detection.width)}×{Math.round(detection.height)} px
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
