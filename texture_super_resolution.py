#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def blur2d(img: np.ndarray) -> np.ndarray:
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    pad = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = np.sum(pad[y : y + 3, x : x + 3] * k)
    return out


def sharpness_score(rgb: np.ndarray) -> float:
    luma = to_luma(rgb)
    smooth = blur2d(luma)
    lap = luma - smooth
    return float(np.var(lap))


def resize_to_target(rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    pil = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def normalize_illumination(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    img_mean = image.mean(axis=(0, 1), keepdims=True)
    img_std = image.std(axis=(0, 1), keepdims=True) + 1e-6
    ref_mean = reference.mean(axis=(0, 1), keepdims=True)
    ref_std = reference.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (image - img_mean) / img_std * ref_std + ref_mean
    return np.clip(normalized, 0, 255)


def compute_softmax_weights(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    temperature = max(float(np.std(scores)), 0.1)
    exp_scores = np.exp(shifted / temperature)
    return exp_scores / (np.sum(exp_scores) + 1e-6)


def robust_fusion(frames: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stack = np.stack(frames, axis=0)  # [N,H,W,C]
    median = np.median(stack, axis=0, keepdims=True)
    deviation = np.abs(stack - median).mean(axis=3, keepdims=True)  # [N,H,W,1]
    consistency = np.exp(-deviation / 10.0)
    w = weights[:, None, None, None] * consistency
    denom = np.sum(w, axis=0, keepdims=False) + 1e-6
    fused = np.sum(stack * w, axis=0) / denom
    return np.clip(fused, 0, 255)


def enhance_texture(
    images: Sequence[np.ndarray],
    top_k: int = 10,
    scale: int = 2,
    detail_strength: float = 1.2,
) -> np.ndarray:
    if not images:
        raise ValueError("No images provided")

    scores = np.array([sharpness_score(img) for img in images], dtype=np.float32)
    order = np.argsort(-scores)
    chosen_idx = order[: max(1, min(top_k, len(images)))]
    chosen = [images[i] for i in chosen_idx]
    chosen_scores = scores[chosen_idx]

    target_h = min(img.shape[0] for img in chosen)
    target_w = min(img.shape[1] for img in chosen)
    chosen = [resize_to_target(img, (target_h, target_w)) for img in chosen]

    ref = chosen[0]
    normalized = [normalize_illumination(img, ref) for img in chosen]

    w = compute_softmax_weights(chosen_scores)

    fused = robust_fusion(normalized, w)
    fused_pil = Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8), mode="RGB")

    if scale > 1:
        fused_pil = fused_pil.resize(
            (fused_pil.width * scale, fused_pil.height * scale),
            resample=Image.Resampling.BICUBIC,
        )

    # Unsharp mask for light detail recovery.
    radius = 1.0
    percent = min(500, int(max(0.0, detail_strength) * 100))
    fused_pil = fused_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))

    return np.asarray(fused_pil, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse multiple facial images into a super-resolved texture map."
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with face images")
    p.add_argument("--output", type=Path, required=True, help="Output texture image path")
    p.add_argument("--top-k", type=int, default=10, help="How many sharpest images to fuse")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor after fusion")
    p.add_argument(
        "--detail-strength",
        type=float,
        default=1.2,
        help="Unsharp mask strength (0.0 disables extra sharpening)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    paths = list_images(args.input_dir)
    if not paths:
        raise ValueError(f"No valid image files found in: {args.input_dir}")

    images = [load_rgb(p) for p in paths]
    sr = enhance_texture(
        images=images,
        top_k=max(1, args.top_k),
        scale=max(1, args.scale),
        detail_strength=max(0.0, args.detail_strength),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sr, mode="RGB").save(args.output)
    print(f"Saved enhanced texture to: {args.output}")


if __name__ == "__main__":
    main()
