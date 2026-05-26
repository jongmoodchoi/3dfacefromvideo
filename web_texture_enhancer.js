const MIN_TEMPERATURE = 0.1;
const CONSISTENCY_SCALE = 10.0;

function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function toImageData(source, targetWidth, targetHeight) {
  const canvas = createCanvas(targetWidth, targetHeight);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, targetWidth, targetHeight);
  return ctx.getImageData(0, 0, targetWidth, targetHeight);
}

function computeLuma(data, width, height) {
  const out = new Float32Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = 0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2];
  }
  return out;
}

function blur3x3Reflect(input, width, height) {
  const out = new Float32Array(input.length);
  const k = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let idx = 0;
      for (let ky = -1; ky <= 1; ky++) {
        const ry = Math.max(0, Math.min(height - 1, y + ky));
        for (let kx = -1; kx <= 1; kx++) {
          const rx = Math.max(0, Math.min(width - 1, x + kx));
          sum += input[ry * width + rx] * k[idx++];
        }
      }
      out[y * width + x] = sum / 16;
    }
  }
  return out;
}

function sharpnessScore(imageData) {
  const { data, width, height } = imageData;
  const luma = computeLuma(data, width, height);
  const smooth = blur3x3Reflect(luma, width, height);
  let mean = 0;
  const lap = new Float32Array(luma.length);
  for (let i = 0; i < luma.length; i++) {
    lap[i] = luma[i] - smooth[i];
    mean += lap[i];
  }
  mean /= lap.length;
  let variance = 0;
  for (let i = 0; i < lap.length; i++) {
    const d = lap[i] - mean;
    variance += d * d;
  }
  return variance / lap.length;
}

function computeStats(data) {
  const n = data.length / 4;
  const mean = [0, 0, 0];
  const std = [0, 0, 0];
  for (let i = 0; i < data.length; i += 4) {
    mean[0] += data[i];
    mean[1] += data[i + 1];
    mean[2] += data[i + 2];
  }
  mean[0] /= n;
  mean[1] /= n;
  mean[2] /= n;
  for (let i = 0; i < data.length; i += 4) {
    std[0] += (data[i] - mean[0]) ** 2;
    std[1] += (data[i + 1] - mean[1]) ** 2;
    std[2] += (data[i + 2] - mean[2]) ** 2;
  }
  std[0] = Math.sqrt(std[0] / n) + 1e-6;
  std[1] = Math.sqrt(std[1] / n) + 1e-6;
  std[2] = Math.sqrt(std[2] / n) + 1e-6;
  return { mean, std };
}

function normalizeIllumination(imageData, refImageData) {
  const output = new Uint8ClampedArray(imageData.data.length);
  const imageStats = computeStats(imageData.data);
  const refStats = computeStats(refImageData.data);
  for (let i = 0; i < imageData.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const value =
        ((imageData.data[i + c] - imageStats.mean[c]) / imageStats.std[c]) *
          refStats.std[c] +
        refStats.mean[c];
      output[i + c] = Math.max(0, Math.min(255, value));
    }
    output[i + 3] = 255;
  }
  return new ImageData(output, imageData.width, imageData.height);
}

function softmaxWeights(scores) {
  const maxScore = Math.max(...scores);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance =
    scores.reduce((acc, x) => acc + (x - mean) ** 2, 0) / scores.length;
  const std = Math.sqrt(variance);
  const temperature = Math.max(std, MIN_TEMPERATURE);
  const exps = scores.map((s) => Math.exp((s - maxScore) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0) + 1e-6;
  return exps.map((x) => x / sum);
}

function robustFuse(imageDataList, weights) {
  const { width, height } = imageDataList[0];
  const pixelCount = width * height;
  const output = new Uint8ClampedArray(pixelCount * 4);
  const frameCount = imageDataList.length;

  for (let i = 0; i < pixelCount; i++) {
    const base = i * 4;
    for (let c = 0; c < 3; c++) {
      const vals = new Array(frameCount);
      for (let f = 0; f < frameCount; f++) {
        vals[f] = imageDataList[f].data[base + c];
      }
      const sorted = [...vals].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      let num = 0;
      let den = 1e-6;
      for (let f = 0; f < frameCount; f++) {
        const consistency = Math.exp(
          -Math.abs(vals[f] - median) / CONSISTENCY_SCALE
        );
        const w = weights[f] * consistency;
        num += vals[f] * w;
        den += w;
      }
      output[base + c] = Math.max(0, Math.min(255, num / den));
    }
    output[base + 3] = 255;
  }
  return new ImageData(output, width, height);
}

function upscaleAndSharpen(imageData, scale = 2, detailStrength = 1.2) {
  const srcCanvas = createCanvas(imageData.width, imageData.height);
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.putImageData(imageData, 0, 0);

  const dstCanvas = createCanvas(
    imageData.width * Math.max(1, scale),
    imageData.height * Math.max(1, scale)
  );
  const dstCtx = dstCanvas.getContext("2d");
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = "high";
  dstCtx.filter = "none";
  dstCtx.drawImage(srcCanvas, 0, 0, dstCanvas.width, dstCanvas.height);

  const sharpenAmount = Math.max(0, Math.min(3, detailStrength));
  if (sharpenAmount > 0) {
    dstCtx.filter = `contrast(${1 + sharpenAmount * 0.15}) saturate(${
      1 + sharpenAmount * 0.05
    })`;
    dstCtx.drawImage(dstCanvas, 0, 0);
    dstCtx.filter = "none";
  }
  return dstCanvas;
}

async function loadImageSource(fileOrUrl) {
  if (typeof fileOrUrl === "string") {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = fileOrUrl;
    });
  }
  if (fileOrUrl instanceof Blob) {
    const url = URL.createObjectURL(fileOrUrl);
    try {
      const img = await loadImageSource(url);
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  return fileOrUrl;
}

export async function enhanceTextureFromFrames({
  frames,
  topK = 10,
  scale = 2,
  detailStrength = 1.2,
}) {
  if (!frames || !frames.length) {
    throw new Error("frames is required");
  }
  const imageSources = await Promise.all(frames.map(loadImageSource));
  const targetWidth = Math.min(...imageSources.map((s) => s.width));
  const targetHeight = Math.min(...imageSources.map((s) => s.height));
  const imageDataList = imageSources.map((s) =>
    toImageData(s, targetWidth, targetHeight)
  );

  const scored = imageDataList
    .map((img, idx) => ({ idx, score: sharpnessScore(img) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, Math.min(topK, imageDataList.length)));

  const chosen = scored.map((x) => imageDataList[x.idx]);
  const scores = scored.map((x) => x.score);
  const reference = chosen[0];
  const normalized = chosen.map((img) => normalizeIllumination(img, reference));
  const weights = softmaxWeights(scores);
  const fused = robustFuse(normalized, weights);
  return upscaleAndSharpen(fused, scale, detailStrength);
}

export async function applyEnhancedTextureToFace({
  frames,
  faceMesh,
  material,
  topK = 10,
  scale = 2,
  detailStrength = 1.2,
}) {
  const enhancedCanvas = await enhanceTextureFromFrames({
    frames,
    topK,
    scale,
    detailStrength,
  });

  if (typeof THREE === "undefined") {
    throw new Error("THREE is required for mesh texture update");
  }

  const targetMaterial = material || faceMesh?.material;
  if (!targetMaterial) {
    throw new Error("material or faceMesh.material is required");
  }

  const texture = new THREE.CanvasTexture(enhancedCanvas);
  texture.needsUpdate = true;
  targetMaterial.map = texture;
  targetMaterial.needsUpdate = true;

  return { enhancedCanvas, texture };
}
