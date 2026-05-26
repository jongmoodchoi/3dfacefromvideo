# 3dfacefromvideo

3D face reconstruction from video using multiview geometry.

## Multi-image texture enhancement (super-resolution)

This repository now includes a minimal texture enhancement pipeline that fuses
multiple facial frames into a cleaner high-resolution texture image.

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
pip install -r /absolute/path/to/requirements.txt
```

```bash
python3 texture_super_resolution.py \
  --input-dir /absolute/path/to/faces \
  --output /absolute/path/to/texture_sr.png \
  --top-k 10 \
  --scale 2
```

### Input assumptions

- Face crops are already roughly aligned (same identity, similar pose center)
- Images are in `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.webp`

### Output

- A super-resolved texture image (PNG by default based on output extension)
