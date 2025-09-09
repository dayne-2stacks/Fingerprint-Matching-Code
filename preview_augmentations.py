#!/usr/bin/env python3
"""
Preview fingerprint augmentations: applies each augmentation individually and
produces a tiled image showing the results. Optionally overlays keypoints if a
matching annotation file is found next to the input image (.tsv/.csv/.txt).

Usage:
  python scripts/preview_augmentations.py /path/to/fingerprint.jpg \
      --out preview.jpg --seed 123

Annotation formats (same stem as image):
  - .tsv with headers: x\ty
  - .csv with headers: x,y
  - .txt per-line: x,y (no header)
"""

import argparse
from pathlib import Path
import csv
import math
import random
import sys
from typing import List, Tuple

import cv2
import numpy as np

try:
    from utils.augmentation import apply_single_transform, transforms as AUG_TRANSFORMS
except Exception as e:
    print(f"Failed to import augmentation utilities: {e}")
    sys.exit(1)


Anno = List[Tuple[str, float, float]]


def read_keypoints(image_path: Path) -> Anno:
    """Read keypoints from a sibling .tsv/.csv/.txt file if present.

    Returns a list of [label, x, y]. Labels auto-increment when not provided.
    """
    possible_exts = [".tsv", ".csv", ".txt"]
    for ext in possible_exts:
        cand = image_path.with_suffix(ext)
        if cand.exists():
            try:
                if ext == ".txt":
                    ann = []
                    with open(cand, "r") as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            xs, ys = line.split(",")
                            ann.append([str(i), float(xs), float(ys)])
                    return ann
                else:
                    delim = "\t" if ext == ".tsv" else ","
                    ann = []
                    with open(cand, "r") as f:
                        reader = csv.DictReader(f, delimiter=delim)
                        for i, row in enumerate(reader):
                            x = float(row["x"])
                            y = float(row["y"])
                            ann.append([str(i), x, y])
                    return ann
            except Exception as e:
                print(f"Warning: failed to read annotations from {cand}: {e}")
                return []
    return []


def standardize_image_and_ann(img: np.ndarray, ann: Anno) -> Tuple[np.ndarray, Anno]:
    """Resize to 320x320 and center crop to 240x320; adjust keypoints."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
    sx, sy = 320 / w, 320 / h
    ann2 = [[i, x * sx, y * sy] for i, x, y in ann]
    crop_h, crop_w = 240, 320
    start_x = (320 - crop_w) // 2
    start_y = (320 - crop_h) // 2
    cropped = resized[start_y:start_y + crop_h, start_x:start_x + crop_w]
    ann3 = [
        [i, x - start_x, y - start_y]
        for i, x, y in ann2
        if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h
    ]
    return cropped, ann3


def draw_keypoints(img: np.ndarray, ann: Anno, color=(0, 255, 0), radius: int = 3, labels: bool = False) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lid, x, y in ann:
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, int(max(1, radius)), color, -1, lineType=cv2.LINE_AA)
        if labels:
            cv2.putText(out, str(lid), (center[0] + 3, center[1] - 3), font, 0.35, color, 1, cv2.LINE_AA)
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, th + pad - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def tile(images: List[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Preview fingerprint augmentations")
    parser.add_argument("image", type=Path, help="Path to fingerprint image (jpg/png)")
    parser.add_argument("--out", type=Path, default=Path("augmentation_preview.jpg"), help="Output preview image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    # Keypoint overlay controls
    parser.add_argument("--no-kpts", action="store_true", help="Do not overlay keypoints (alias of --kpts-off)")
    parser.add_argument("--kpts-off", action="store_true", help="Disable keypoint highlighting")
    parser.add_argument("--kpt-radius", type=int, default=3, help="Keypoint circle radius")
    parser.add_argument("--kpt-color", type=str, default="0,255,0", help="Keypoint color as R,G,B (0-255)")
    parser.add_argument("--kpt-labels", action="store_true", help="Draw keypoint labels")
    parser.add_argument("--list", action="store_true", help="List available transforms and exit")
    args = parser.parse_args()

    if args.list:
        print("Available transforms:")
        for t in AUG_TRANSFORMS:
            print(f"- {t}")
        return

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    img_path: Path = args.image
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(1)

    ann: Anno = read_keypoints(img_path)

    def parse_color(s: str):
        try:
            parts = [int(p) for p in s.split(",")]
            if len(parts) != 3:
                raise ValueError
            return tuple(int(max(0, min(255, v))) for v in parts)
        except Exception:
            print(f"Invalid --kpt-color '{s}', defaulting to 0,255,0")
            return (0, 255, 0)

    kpt_color = parse_color(args.kpt_color)
    kpt_radius = max(1, int(args.kpt_radius))
    draw_labels = bool(args.kpt_labels)
    overlay = not (args.no_kpts or args.kpts_off)

    # First tile: baseline standardized image
    base_img, base_ann = standardize_image_and_ann(img, ann)
    base_vis = draw_keypoints(base_img, base_ann, color=kpt_color, radius=kpt_radius, labels=draw_labels) if (ann and overlay) else base_img
    tiles = [put_label(base_vis, "baseline")] 

    # One tile per transform
    for t in AUG_TRANSFORMS:
        try:
            aug_img, aug_ann = apply_single_transform(img, ann, t)
            if ann and overlay:
                aug_img = draw_keypoints(aug_img, aug_ann, color=kpt_color, radius=kpt_radius, labels=draw_labels)
            tiles.append(put_label(aug_img, t))
        except Exception as e:
            print(f"Transform '{t}' failed: {e}")
            # Use baseline tile as placeholder
            tiles.append(put_label(base_vis, f"{t} (err)"))

    cols = 3 if len(tiles) <= 9 else 4
    mosaic = tile(tiles, cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), mosaic)
    print(f"Saved preview to: {args.out}")


if __name__ == "__main__":
    main()
