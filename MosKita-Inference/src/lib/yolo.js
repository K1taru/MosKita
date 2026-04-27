function inferTensorShape(dims) {
  if (!Array.isArray(dims)) {
    throw new Error('Tensor dims are required.');
  }

  if (dims.length === 2) {
    return dims;
  }

  if (dims.length === 3) {
    return dims.slice(1);
  }

  throw new Error(`Unsupported output dims: ${dims.join('x')}`);
}

export function resolveOutputShape(dims) {
  const [first, second] = inferTensorShape(dims);
  const firstLooksLikeChannels = first >= 5 && first <= 512;
  const secondLooksLikeChannels = second >= 5 && second <= 512;

  if (firstLooksLikeChannels && !secondLooksLikeChannels) {
    return { rows: second, channels: first, layout: 'channels-first' };
  }

  if (!firstLooksLikeChannels && secondLooksLikeChannels) {
    return { rows: first, channels: second, layout: 'channels-last' };
  }

  if (first <= second) {
    return { rows: second, channels: first, layout: 'channels-first' };
  }

  return { rows: first, channels: second, layout: 'channels-last' };
}

function createAccessor(data, dims) {
  const { rows, channels, layout } = resolveOutputShape(dims);

  if (layout === 'channels-first') {
    return {
      rows,
      channels,
      getValue(rowIndex, channelIndex) {
        return data[channelIndex * rows + rowIndex];
      },
    };
  }

  return {
    rows,
    channels,
    getValue(rowIndex, channelIndex) {
      return data[rowIndex * channels + channelIndex];
    },
  };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function toProbability(rawScore) {
  if (!Number.isFinite(rawScore)) {
    return 0;
  }

  if (rawScore >= 0 && rawScore <= 1) {
    return rawScore;
  }

  return sigmoid(rawScore);
}

function isLikelyNmsOutput(accessor, classNameCount) {
  const { rows, channels, getValue } = accessor;
  if (channels !== 6 || rows <= 0 || classNameCount <= channels - 4) {
    return false;
  }

  const sampleSize = Math.min(rows, 64);
  let confidenceLike = 0;
  let classIdLike = 0;

  for (let sampleIndex = 0; sampleIndex < sampleSize; sampleIndex += 1) {
    const rowIndex = Math.floor((sampleIndex * rows) / sampleSize);
    const confidence = getValue(rowIndex, 4);
    const classId = getValue(rowIndex, 5);

    if (Number.isFinite(confidence) && confidence >= 0 && confidence <= 1.25) {
      confidenceLike += 1;
    }

    const roundedClass = Math.round(classId);
    const looksLikeClassId = Number.isFinite(classId)
      && classId >= -0.5
      && classId <= Math.max(0.5, classNameCount - 0.5)
      && Math.abs(classId - roundedClass) <= 0.25;

    if (looksLikeClassId) {
      classIdLike += 1;
    }
  }

  return confidenceLike / sampleSize >= 0.8 && classIdLike / sampleSize >= 0.8;
}

function decodeNmsOutput(
  accessor,
  {
    classNames,
    confidenceThreshold,
    iouThreshold,
    letterbox,
  },
) {
  const { rows, getValue } = accessor;
  const {
    scale = 1,
    padX = 0,
    padY = 0,
    originalWidth = letterbox.inputSize ?? 640,
    originalHeight = letterbox.inputSize ?? 640,
  } = letterbox;

  const candidates = [];

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const score = clamp(toProbability(getValue(rowIndex, 4)), 0, 1);
    if (score < confidenceThreshold) {
      continue;
    }

    const classId = Math.round(getValue(rowIndex, 5));
    if (classId < 0 || classId >= classNames.length) {
      continue;
    }

    const x1 = getValue(rowIndex, 0);
    const y1 = getValue(rowIndex, 1);
    const x2 = getValue(rowIndex, 2);
    const y2 = getValue(rowIndex, 3);
    if (![x1, y1, x2, y2].every(Number.isFinite)) {
      continue;
    }

    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const right = Math.max(x1, x2);
    const bottom = Math.max(y1, y2);

    const projectedX = (left - padX) / scale;
    const projectedY = (top - padY) / scale;
    const projectedWidth = (right - left) / scale;
    const projectedHeight = (bottom - top) / scale;

    const x = clamp(projectedX, 0, originalWidth);
    const y = clamp(projectedY, 0, originalHeight);
    const maxWidth = Math.max(0, originalWidth - x);
    const maxHeight = Math.max(0, originalHeight - y);
    const clippedWidth = clamp(projectedWidth, 0, maxWidth);
    const clippedHeight = clamp(projectedHeight, 0, maxHeight);

    if (!clippedWidth || !clippedHeight) {
      continue;
    }

    candidates.push({
      x,
      y,
      width: clippedWidth,
      height: clippedHeight,
      score,
      classId,
      className: classNames[classId],
    });
  }

  return applyNms(candidates, iouThreshold).sort((left, right) => right.score - left.score);
}

export function intersectionOverUnion(left, right) {
  const leftX = Math.max(left.x, right.x);
  const topY = Math.max(left.y, right.y);
  const rightX = Math.min(left.x + left.width, right.x + right.width);
  const bottomY = Math.min(left.y + left.height, right.y + right.height);

  const intersectionWidth = Math.max(0, rightX - leftX);
  const intersectionHeight = Math.max(0, bottomY - topY);
  const intersectionArea = intersectionWidth * intersectionHeight;

  if (!intersectionArea) {
    return 0;
  }

  const unionArea = left.width * left.height + right.width * right.height - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

export function applyNms(detections, iouThreshold) {
  const queued = [...detections].sort((left, right) => right.score - left.score);
  const retained = [];

  while (queued.length) {
    const candidate = queued.shift();
    retained.push(candidate);

    for (let index = queued.length - 1; index >= 0; index -= 1) {
      const other = queued[index];
      const sameClass = candidate.classId === other.classId;
      if (sameClass && intersectionOverUnion(candidate, other) >= iouThreshold) {
        queued.splice(index, 1);
      }
    }
  }

  return retained;
}

export function decodeYoloOutput(
  outputTensor,
  {
    classNames,
    confidenceThreshold = 0.4,
    iouThreshold = 0.45,
    letterbox = {},
  },
) {
  const { data, dims } = outputTensor ?? {};
  if (!data || !dims) {
    return [];
  }

  const accessor = createAccessor(data, dims);
  const { rows, channels, getValue } = accessor;

  if (isLikelyNmsOutput(accessor, classNames.length)) {
    return decodeNmsOutput(accessor, {
      classNames,
      confidenceThreshold,
      iouThreshold,
      letterbox,
    });
  }

  const hasObjectness = channels - 5 === classNames.length;
  const classOffset = hasObjectness ? 5 : 4;
  const objectnessOffset = hasObjectness ? 4 : -1;
  const classCount = channels - classOffset;
  const resolvedClassCount = Math.min(classNames.length, classCount);
  if (resolvedClassCount <= 0) {
    return [];
  }

  const {
    scale = 1,
    padX = 0,
    padY = 0,
    originalWidth = letterbox.inputSize ?? 640,
    originalHeight = letterbox.inputSize ?? 640,
  } = letterbox;

  const candidates = [];

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const objectnessScore = objectnessOffset >= 0
      ? toProbability(getValue(rowIndex, objectnessOffset))
      : 1;

    let bestScore = 0;
    let bestClassIndex = -1;

    for (let classIndex = 0; classIndex < resolvedClassCount; classIndex += 1) {
      const classScore = toProbability(getValue(rowIndex, classIndex + classOffset));
      const score = objectnessScore * classScore;
      if (score > bestScore) {
        bestScore = score;
        bestClassIndex = classIndex;
      }
    }

    if (bestScore < confidenceThreshold || bestClassIndex === -1) {
      continue;
    }

    const centerX = getValue(rowIndex, 0);
    const centerY = getValue(rowIndex, 1);
    const width = getValue(rowIndex, 2);
    const height = getValue(rowIndex, 3);

    const projectedX = (centerX - width / 2 - padX) / scale;
    const projectedY = (centerY - height / 2 - padY) / scale;
    const projectedWidth = width / scale;
    const projectedHeight = height / scale;

    const x = clamp(projectedX, 0, originalWidth);
    const y = clamp(projectedY, 0, originalHeight);
    const maxWidth = Math.max(0, originalWidth - x);
    const maxHeight = Math.max(0, originalHeight - y);
    const clippedWidth = clamp(projectedWidth, 0, maxWidth);
    const clippedHeight = clamp(projectedHeight, 0, maxHeight);

    if (!clippedWidth || !clippedHeight) {
      continue;
    }

    candidates.push({
      x,
      y,
      width: clippedWidth,
      height: clippedHeight,
      score: bestScore,
      classId: bestClassIndex,
      className: classNames[bestClassIndex],
    });
  }

  return applyNms(candidates, iouThreshold).sort((left, right) => right.score - left.score);
}