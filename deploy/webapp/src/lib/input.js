import * as ort from 'onnxruntime-web';

export function createInputTensor(source, { inputSize = 640, canvas } = {}) {
  const workingCanvas = canvas ?? document.createElement('canvas');
  const context = workingCanvas.getContext('2d', { willReadFrequently: true });

  const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
  const sourceHeight = source.videoHeight || source.naturalHeight || source.height;

  if (!sourceWidth || !sourceHeight) {
    throw new Error('Source dimensions are not ready yet.');
  }

  workingCanvas.width = inputSize;
  workingCanvas.height = inputSize;
  context.fillStyle = 'rgb(114, 114, 114)';
  context.fillRect(0, 0, inputSize, inputSize);

  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
  const resizedWidth = Math.round(sourceWidth * scale);
  const resizedHeight = Math.round(sourceHeight * scale);
  const padX = (inputSize - resizedWidth) / 2;
  const padY = (inputSize - resizedHeight) / 2;

  context.drawImage(source, 0, 0, sourceWidth, sourceHeight, padX, padY, resizedWidth, resizedHeight);

  const { data } = context.getImageData(0, 0, inputSize, inputSize);
  const channelLength = inputSize * inputSize;
  const tensorData = new Float32Array(channelLength * 3);

  for (let pixelIndex = 0; pixelIndex < channelLength; pixelIndex += 1) {
    const rgbaOffset = pixelIndex * 4;
    tensorData[pixelIndex] = data[rgbaOffset] / 255;
    tensorData[channelLength + pixelIndex] = data[rgbaOffset + 1] / 255;
    tensorData[channelLength * 2 + pixelIndex] = data[rgbaOffset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]),
    letterbox: {
      scale,
      padX,
      padY,
      inputSize,
      originalWidth: sourceWidth,
      originalHeight: sourceHeight,
    },
  };
}