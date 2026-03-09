# Fingerprint Matching Code

This repository provides utilities for fingerprint matching and a simple classification baseline.

## Dataset Layout
- Images: `dataset/Synthetic/R1` – `R5` with `.jpg` files.
- Keypoints: `.tsv` files next to each image containing `x` and `y` columns.
  Each keypoint entry is given a unique label combining the folder, file name
  and its index so labels do not collide across images.

## Classification Baseline
The classification task reuses the same dataset structure. Genuine pairs are created by duplicating a single image and applying independent augmentations. Imposter pairs come from different fingers and use a zero permutation matrix. The loader handles augmentation internally.

## Training Plan (Dustbin Sinkhorn + AFAT)
The matcher now learns with three complementary losses:
- **Permutation loss** on the refined real<->real block (AFAT top-k mask x Sinkhorn).
- **Dustbin supervision** on the dustbin row/column to explicitly learn outliers.
- **k regression loss** (optional) for AFAU when `regression=True`.

### Proposed 3-Stage Schedule
**Stage 1: Matcher warm-up**
- Train CNN + GM stack: `node_layers`, `edge_layers`, `final_layers`, `message_pass_node_features`,
  `build_edge_features_from_node_features`, `vertex_affinity`, `edge_affinity`, `gnn_layer_*`, `classifier`,
  and `dustbin_bias_src/dustbin_bias_tgt`.
- Use GT `k` (from the GT permutation block).
- Loss: permutation + dustbin supervision.
- *Optional:* freeze AFAU modules (`encoder_k`, `final_row`, `final_col`) for stability.

**Stage 2: AFAU warm-up**
- Freeze CNN + GM stack.
- Train only AFAU: `encoder_k`, `final_row`, `final_col`.
- Loss: permutation + dustbin + k-regression (`ks_loss`).

**Stage 3: Joint fine-tuning**
- Unfreeze everything (matcher + AFAU).
- Loss: permutation + dustbin + k-regression.

### Running Training
Single-stage (simpler, uses permutation + dustbin; k-regression only if enabled):
```bash
python train.py
```

Three-stage schedule (recommended when using k-regression):
```bash
python train_new.py
```

## Evaluating the Binary Matcher

Use `evaluate_binary_classifier.py` to compute verification metrics for the
matcher. The script loads the trained network from
`results/base_w_k_dustbin/params/best_model.pt` and scores each pair using the
ratio `k_pred / min_points`, where `k_pred` is the number of predicted matches
and `min_points` is the minimum keypoints in the pair.

The evaluation writes `metrics.csv`, an `eval.log` file with the values, and
ROC/PR curve images to the `results/base_w_k_dustbin/<dataset>` directory.

Run the evaluation with:

```bash
python evaluate_binary_classifier.py
```



# Running Docker File

docker build -t fingerprint-thinkmatch .

docker run --gpus all -it \
  --rm \
  fingerprint-thinkmatch
