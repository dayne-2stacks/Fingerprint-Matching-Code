# Fingerprint Matching Code

This repository provides utilities for fingerprint matching and a simple classification baseline.

## Dataset Layout
- Images: `dataset/Synthetic/R1` – `R5` with `.jpg` files.
- Keypoints: `.tsv` files next to each image containing `x` and `y` columns.
  Each keypoint entry is given a unique label combining the folder, file name
  and its index so labels do not collide across images.

## Classification Baseline
The classification task reuses the same dataset structure. Genuine pairs are created by duplicating a single image and applying independent augmentations. Imposter pairs come from different fingers and use a zero permutation matrix. The loader handles augmentation internally.
