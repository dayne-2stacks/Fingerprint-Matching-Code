# Fingerprint Matching Code

This repository provides utilities for fingerprint matching and a simple classification baseline.

## Dataset Layout
- Images: `dataset/Synthetic/R1` – `R5` with `.jpg` files.
- Keypoints: `.tsv` files next to each image containing `x` and `y` columns.
  Each keypoint entry is given a unique label combining the folder, file name
  and its index so labels do not collide across images.

## Classification Baseline
The classification task reuses the same dataset structure. Genuine pairs are created by duplicating a single image and applying independent augmentations. Imposter pairs come from different fingers and use a zero permutation matrix. The loader handles augmentation internally.

## Evaluating the Binary Classifier

Use `evaluate_binary_classifier.py` to compute verification metrics for the
binary classification model. The script loads the trained network from
`results/binary-classifier/params/best_model.pt` and determines whether two
fingerprints match based on the predicted `k` value: if `k` equals zero the pair
is classified as an imposter.

The evaluation writes `metrics.csv`, an `eval.log` file with the values,
and ROC/PR curve images to the `results/binary-classifier` directory.

Run the evaluation with:

```bash
python evaluate_binary_classifier.py
```
