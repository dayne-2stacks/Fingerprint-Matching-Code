#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import random

import cv2
import numpy as np

from utils.augmentation import apply_single_transform, transforms as AUG_TRANSFORMS


def read_annotations(image_path: Path, sample_stride: int = 20):
    """Read annotations from sibling .tsv/.csv/.txt and subsample for clarity."""
    for ext in (".tsv", ".csv", ".txt"):
        ann_path = image_path.with_suffix(ext)
        if not ann_path.exists():
            continue
        annotations = []
        with ann_path.open("r") as f:
            for i, line in enumerate(f):
                if i % max(1, sample_stride) != 0:
                    continue
                parts = line.strip().replace(",", "\t").split("\t")
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    annotations.append([i, x, y])
                except ValueError:
                    continue
        if annotations:
            return annotations
    return []


def draw_annotations(image, annotations, radius=3, color=(0, 255, 0)):
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()
    for idx, x, y in annotations:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(annotated, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            annotated,
            str(idx),
            (cx + radius + 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def add_tile_label(image, label):
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 5
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, label, (pad, th + pad - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def make_grid(images, cols):
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def pick_default_image():
    candidates = sorted(Path("dataset/Synthetic/R1").glob("*.jpg"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No default image found in dataset/Synthetic/R1. Pass --image.")


def main():
    parser = argparse.ArgumentParser(description="Show all fingerprint transforms side by side.")
    parser.add_argument("--image", type=Path, default=None, help="Fingerprint image path")
    parser.add_argument("--out", type=Path, default=Path("output/all_transforms_side_by_side.jpg"), help="Output preview image path")
    parser.add_argument("--cols", type=int, default=4, help="Number of grid columns")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--sample-stride", type=int, default=1, help="Subsample annotation rows")
    parser.add_argument("--no-annotations", action="store_true", help="Disable annotation drawing")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    image_path = args.image if args.image is not None else pick_default_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    annotations = read_annotations(image_path, sample_stride=args.sample_stride)
    print(f"Image: {image_path}")
    print(f"Loaded {len(annotations)} annotations")

    tiles = []

    # "baseline" uses unknown transform key -> no-op + standardize to model geometry.
    base_img, base_ann = apply_single_transform(image, annotations, transformation_type="baseline")
    if not args.no_annotations and annotations:
        base_img = draw_annotations(base_img, base_ann)
    tiles.append(add_tile_label(base_img, "baseline"))

    for transform_name in AUG_TRANSFORMS:
        try:
            aug_img, aug_ann = apply_single_transform(image, annotations, transform_name)
            if not args.no_annotations and annotations:
                aug_img = draw_annotations(aug_img, aug_ann)
            tiles.append(add_tile_label(aug_img, transform_name))
        except Exception as exc:
            failed = add_tile_label(base_img, f"{transform_name} (err)")
            cv2.putText(
                failed,
                str(exc)[:48],
                (6, failed.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(failed)

    mosaic = make_grid(tiles, cols=max(1, args.cols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out), mosaic)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.out}")
    print(f"Saved side-by-side preview to: {args.out}")
    print(f"Transforms shown: baseline + {len(AUG_TRANSFORMS)} augmentations")


if __name__ == "__main__":
    main()
