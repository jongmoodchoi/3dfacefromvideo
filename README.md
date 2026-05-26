# 3dfacefromvideo

3D face reconstruction from video using multiview geometry.

## Multi-image texture enhancement (super-resolution)

This repository now includes a minimal texture enhancement pipeline that fuses
multiple facial frames into a cleaner high-resolution texture image.

## Web integration for 3D reconstruction result texture

The browser module can update the final 3D face texture directly from multiple
facial frames:

- `web_texture_enhancer.js`: browser-side multi-frame texture enhancement
- `index.html`: minimal demo UI (file upload -> enhanced texture preview/apply)

If your web reconstruction flow has a face mesh (e.g. Three.js mesh), expose it
as `window.faceMesh` and open `index.html`. The generated enhanced texture will
be applied to `faceMesh.material.map`.

For custom integration:

1. Import `applyEnhancedTextureToFace` from `web_texture_enhancer.js`
2. Pass multiple face frames (`File`, `Blob`, image URL, or HTMLImageElement)
3. Pass `faceMesh` (or `material`) and tuning params (`topK`, `scale`, `detailStrength`)

### What it does

- Loads a sequence of face images from a directory
- Scores frame sharpness and selects the best frames
- Normalizes illumination across selected frames
- Fuses frames with a robust weighted strategy
- Upscales the fused texture map
- Applies a light detail enhancement step

### Files

- `texture_super_resolution.py`: CLI tool and fusion pipeline implementation

### Usage

```bash
pip install -r requirements.txt
```

```bash
python3 texture_super_resolution.py \
  --input-dir ./faces \
  --output ./output/texture_sr.png \
  --top-k 10 \
  --scale 2
```

### Input assumptions

- Face crops are already roughly aligned (same identity, similar pose center)
- Images are in `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.webp`

### Output

- A super-resolved texture image (PNG by default based on output extension)
