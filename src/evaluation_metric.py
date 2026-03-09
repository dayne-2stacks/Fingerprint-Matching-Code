import torch
from torch import Tensor
from itertools import combinations
from typing import Tuple
from utils.hungarian import hungarian


def pck(x: Tensor, x_gt: Tensor, perm_mat: Tensor, dist_threshs: Tensor, ns: Tensor) -> Tensor:
    r"""
    Percentage of Correct Keypoints (PCK) evaluation metric.

    If the distance between predicted keypoint and the ground truth keypoint is smaller than a given threshold, than it
    is regraded as a correct matching.

    This is the evaluation metric used by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    :param x: :math:`(b\times n \times 2)` candidate coordinates. :math:`n`: number of nodes in input graph
    :param x_gt: :math:`(b\times n_{gt} \times 2)` ground truth coordinates. :math:`n_{gt}`: number of nodes in ground
     truth graph
    :param perm_mat: :math:`(b\times n \times n_{gt})` permutation matrix or doubly-stochastic matrix indicating
     node-to-node correspondence
    :param dist_threshs: :math:`(b\times m)` a tensor contains thresholds in pixel. :math:`m`: number of thresholds for
     each batch
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(m)` the PCK values of this batch

    .. note::
        An example of ``dist_threshs`` for 4 batches and 2 thresholds:
        ::

            [[10, 20],
             [10, 20],
             [10, 20],
             [10, 20]]
    """
    device = x.device
    batch_num = x.shape[0]
    thresh_num = dist_threshs.shape[1]

    perm_mat = perm_mat[..., :x.shape[1], :x_gt.shape[1]]
    indices = torch.argmax(perm_mat, dim=-1)

    dist = torch.zeros(batch_num, x_gt.shape[1], device=device)
    for b in range(batch_num):
        x_correspond = x[b, indices[b], :]
        dist[b, 0:ns[b]] = torch.norm(x_correspond - x_gt[b], p=2, dim=-1)[0:ns[b]]

    match_num = torch.zeros(thresh_num, device=device)
    total_num = torch.zeros(thresh_num, device=device)
    for b in range(batch_num):
        for idx in range(thresh_num):
            matches = (dist[b] < dist_threshs[b, idx])[0:ns[b]]
            match_num[idx] += torch.sum(matches).to(match_num.dtype)
            total_num[idx] += ns[b].to(total_num.dtype)

    return match_num / total_num


def _ns_value(ns_entry, batch_idx: int, default_value: int) -> int:
    if ns_entry is None:
        return int(default_value)
    if isinstance(ns_entry, Tensor):
        flat = ns_entry.reshape(-1)
        if flat.numel() == 0:
            return int(default_value)
        idx = min(int(batch_idx), flat.numel() - 1)
        return int(flat[idx].item())
    if isinstance(ns_entry, (list, tuple)):
        if len(ns_entry) == 0:
            return int(default_value)
        idx = min(int(batch_idx), len(ns_entry) - 1)
        return int(ns_entry[idx])
    try:
        return int(ns_entry)
    except (TypeError, ValueError):
        return int(default_value)


def _resolve_ns_pair(ns, batch_idx: int, default_rows: int, default_cols: int):
    row_entry = None
    col_entry = None

    if isinstance(ns, (list, tuple)):
        if len(ns) >= 2:
            row_entry, col_entry = ns[0], ns[1]
        elif len(ns) == 1:
            row_entry = col_entry = ns[0]
    elif isinstance(ns, Tensor):
        if ns.ndim == 2:
            if ns.shape[0] >= 2:
                row_entry, col_entry = ns[0], ns[1]
            elif ns.shape[1] >= 2:
                row_entry, col_entry = ns[:, 0], ns[:, 1]
            else:
                row_entry = col_entry = ns.reshape(-1)
        else:
            row_entry = col_entry = ns
    else:
        row_entry = col_entry = ns

    n_rows = _ns_value(row_entry, batch_idx, default_rows)
    n_cols = _ns_value(col_entry, batch_idx, default_cols)
    return max(n_rows, 0), max(n_cols, 0)


def _has_dustbin_row_or_col(mat: Tensor) -> bool:
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        return False
    return bool((mat[-1, :].sum() > 1) or (mat[:, -1].sum() > 1))


def _is_binary_permutation_matrix(mat: Tensor) -> bool:
    if mat.ndim != 2:
        return False
    if not bool(torch.all((mat == 0) | (mat == 1))):
        return False
    row_ok = bool(torch.all(torch.sum(mat, dim=-1) <= 1))
    col_ok = bool(torch.all(torch.sum(mat, dim=-2) <= 1))
    return row_ok and col_ok


def _align_and_strip_dustbin(
    pmat_pred: Tensor,
    pmat_gt: Tensor,
    ns,
    batch_idx: int,
) -> Tuple[Tensor, Tensor]:
    pred_b = pmat_pred[batch_idx]
    gt_b = pmat_gt[batch_idx]

    default_rows = min(pred_b.shape[0], gt_b.shape[0])
    default_cols = min(pred_b.shape[1], gt_b.shape[1])
    n_rows, n_cols = _resolve_ns_pair(ns, batch_idx, default_rows, default_cols)

    row_end = min(max(n_rows, 0), pred_b.shape[0], gt_b.shape[0])
    col_end = min(max(n_cols, 0), pred_b.shape[1], gt_b.shape[1])
    pred_b = pred_b[:row_end, :col_end]
    gt_b = gt_b[:row_end, :col_end]

    common_rows = min(pred_b.shape[0], gt_b.shape[0])
    common_cols = min(pred_b.shape[1], gt_b.shape[1])
    pred_b = pred_b[:common_rows, :common_cols]
    gt_b = gt_b[:common_rows, :common_cols]

    if _has_dustbin_row_or_col(pred_b) or _has_dustbin_row_or_col(gt_b):
        pred_b = pred_b[:-1, :-1]
        gt_b = gt_b[:-1, :-1]

    return pred_b, gt_b


def _f1_from_precision_recall(precision: Tensor, recall: Tensor) -> Tensor:
    denom = precision + recall
    return torch.where(denom > 0, (2 * precision * recall) / denom, torch.zeros_like(denom))


def matching_metrics_from_counts(tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> dict:
    tp = tp.to(dtype=torch.float32)
    tn = tn.to(dtype=torch.float32)
    fp = fp.to(dtype=torch.float32)
    fn = fn.to(dtype=torch.float32)

    total = tp + tn + fp + fn
    pred_pos = tp + fp
    gt_pos = tp + fn
    pred_neg = tn + fn
    gt_neg = tn + fp

    accuracy = torch.where(total > 0, (tp + tn) / total, torch.ones_like(total))

    precision_pos = torch.where(pred_pos > 0, tp / pred_pos, (gt_pos == 0).to(tp.dtype))
    recall_pos = torch.where(gt_pos > 0, tp / gt_pos, (pred_pos == 0).to(tp.dtype))
    f1_pos = _f1_from_precision_recall(precision_pos, recall_pos)

    precision_neg = torch.where(pred_neg > 0, tn / pred_neg, (gt_neg == 0).to(tp.dtype))
    recall_neg = torch.where(gt_neg > 0, tn / gt_neg, (pred_neg == 0).to(tp.dtype))
    f1_neg = _f1_from_precision_recall(precision_neg, recall_neg)

    macro_f1 = (f1_pos + f1_neg) / 2.0

    micro_tp = tp + tn
    micro_fp = fp + fn
    micro_fn = fn + fp
    micro_pred_pos = micro_tp + micro_fp
    micro_gt_pos = micro_tp + micro_fn
    micro_precision = torch.where(
        micro_pred_pos > 0, micro_tp / micro_pred_pos, torch.ones_like(micro_pred_pos)
    )
    micro_recall = torch.where(
        micro_gt_pos > 0, micro_tp / micro_gt_pos, torch.ones_like(micro_gt_pos)
    )
    micro_f1 = _f1_from_precision_recall(micro_precision, micro_recall)

    return {
        "accuracy": accuracy,
        "precision": precision_pos,
        "recall": recall_pos,
        "f1": f1_pos,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class_precision": torch.stack((precision_neg, precision_pos), dim=-1),
        "per_class_recall": torch.stack((recall_neg, recall_pos), dim=-1),
        "per_class_f1": torch.stack((f1_neg, f1_pos), dim=-1),
    }


def matching_classification_metrics(pmat_pred: Tensor, pmat_gt: Tensor, ns) -> dict:
    """Compute binary matching metrics with dustbin-aware cropping."""
    device = pmat_pred.device
    pmat_gt = pmat_gt.to(device)
    batch_num = pmat_pred.shape[0]

    tp = torch.zeros(batch_num, device=device, dtype=torch.float32)
    tn = torch.zeros(batch_num, device=device, dtype=torch.float32)
    fp = torch.zeros(batch_num, device=device, dtype=torch.float32)
    fn = torch.zeros(batch_num, device=device, dtype=torch.float32)

    for b in range(batch_num):
        pred_b, gt_b = _align_and_strip_dustbin(pmat_pred, pmat_gt, ns, b)

        if pred_b.numel() == 0 or gt_b.numel() == 0:
            continue

        if not _is_binary_permutation_matrix(pred_b):
            pred_b = hungarian(pred_b)

        pred_bin = (pred_b > 0.5).to(dtype=torch.bool)
        gt_bin = (gt_b > 0.5).to(dtype=torch.bool)

        tp[b] = torch.sum(pred_bin & gt_bin).to(dtype=torch.float32)
        tn[b] = torch.sum((~pred_bin) & (~gt_bin)).to(dtype=torch.float32)
        fp[b] = torch.sum(pred_bin & (~gt_bin)).to(dtype=torch.float32)
        fn[b] = torch.sum((~pred_bin) & gt_bin).to(dtype=torch.float32)

    derived = matching_metrics_from_counts(tp, tn, fp, fn)
    derived.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return derived


def matching_recall(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    return matching_classification_metrics(pmat_pred, pmat_gt, ns)["recall"]


def matching_precision(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    return matching_classification_metrics(pmat_pred, pmat_gt, ns)["precision"]


def matching_recall_varied(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    return matching_recall(pmat_pred, pmat_gt, ns)


def matching_precision_varied(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    return matching_precision(pmat_pred, pmat_gt, ns)


def matching_accuracy(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor, idx: int) -> Tensor:
    return matching_classification_metrics(pmat_pred, pmat_gt, ns)["accuracy"]


def format_accuracy_metric(ps: Tensor, rs: Tensor, f1s: Tensor) -> str:
    r"""
    Helper function for formatting precision, recall and f1 score metric

    :param ps: tensor containing precisions
    :param rs: tensor containing recalls
    :param f1s: tensor containing f1 scores
    :return: a formatted string with mean and variance of precision, recall and f1 score

    Example output:
    ::

        p = 0.7837±0.2799, r = 0.7837±0.2799, f1 = 0.7837±0.2799
    """
    return 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
        .format(torch.mean(ps), torch.std(ps), torch.mean(rs), torch.std(rs), torch.mean(f1s), torch.std(f1s))

def format_metric(ms: Tensor) -> str:
    r"""
    Helping function for formatting single metric.

    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}±{:.4f}'.format(torch.mean(ms), torch.std(ms))


def objective_score(pmat_pred: Tensor, affmtx: Tensor) -> Tensor:
    r"""
    Objective score given predicted permutation matrix and affinity matrix from the problem.

    .. math::
        \text{objective score} = \mathrm{vec}(\mathbf{X})^\top \mathbf{K} \mathrm{vec}(\mathbf{X})

    where :math:`\mathrm{vec}(\cdot)` means column-wise vectorization.

    :param pmat_pred: predicted permutation matrix :math:`(\mathbf{X})`
    :param affmtx: affinity matrix of the quadratic assignment problem :math:`(\mathbf{K})`
    :return: objective scores

    .. note::
        The most general mathematical form of graph matching is known as Quadratic Assignment Problem (QAP), which is an
        NP-hard combinatorial optimization problem. Objective score reflects the power of the graph matching/QAP solver
        concerning the objective score of the QAP.
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score

def clustering_accuracy(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering accuracy for clusters.

    :math:`\mathcal{A}, \mathcal{B}, ...` are ground truth classes and :math:`\mathcal{A}^\prime, \mathcal{B}^\prime,
    ...` are predicted classes and :math:`k` is the number of classes:

    .. math::
        \text{clustering accuracy} = 1 - \frac{1}{k} \left(\sum_{\mathcal{A}} \sum_{\mathcal{A}^\prime \neq \mathcal{B}^\prime}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{B}^\prime \cap \mathcal{A}|}{|\mathcal{A}| |\mathcal{A}|} +
         \sum_{\mathcal{A}^\prime} \sum_{\mathcal{A} \neq \mathcal{B}}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{A}^\prime \cap \mathcal{B}|}{|\mathcal{A}| |\mathcal{B}|} \right)

    This metric is proposed by `"Wang et al. Clustering-aware Multiple Graph Matching via Decayed Pairwise Matching
    Composition. AAAI 2020." <https://ojs.aaai.org/index.php/AAAI/article/view/5528/5384>`_

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering accuracy
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    batch_num = pred_clusters.shape[0]

    gt_classes_t = []

    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_acc = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        sum = 0
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                pred_i = (pred_clusters[b] == i).to(dtype=torch.float)
                gt_j = (gt_clusters[b] == j).to(dtype=torch.float)
                gt_k = (gt_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(pred_i * gt_j) * torch.sum(pred_i * gt_k)) / torch.sum(pred_i) ** 2
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                gt_i = (gt_clusters[b] == i).to(dtype=torch.float)
                pred_j = (pred_clusters[b] == j).to(dtype=torch.float)
                pred_k = (pred_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(gt_i * pred_j) * torch.sum(gt_i * pred_k)) / (torch.sum(pred_j) * torch.sum(pred_k))

        cluster_acc[b] = 1 - sum / num_clusters[b].to(dtype=torch.float)

    return cluster_acc

def clustering_purity(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering purity for clusters.

    :math:`n` is the number of instances,
    :math:`\mathcal{C}_i` represent the predicted class :math:`i` and :math:`\mathcal{C}^{gt}_j` is ground truth class :math:`j`:

    .. math::
        \text{clustering purity} = \frac{1}{n} \sum_{i=1}^{k} \max_{j\in\{1,...,k\}} |\mathcal{C}_i \cap \mathcal{C}^{gt}_{j}|

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_purity = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        for i in range(num_clusters[b]):
            max_counts = torch.max(torch.unique(gt_clusters[b][pred_clusters[b] == i], return_counts=True)[-1]).to(dtype=torch.float)
            cluster_purity[b] += max_counts / num_instances

    return cluster_purity


def rand_index(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Rand index measurement for clusters.

    Rand index is computed by the number of instances predicted in the same class with the same label :math:`n_{11}` and
    the number of instances predicted in separate classes and with different labels :math:`n_{00}`, normalized by the total
    number of instances pairs :math:`n(n-1)`:

    .. math::
        \text{rand index} = \frac{n_{11} + n_{00}}{n(n-1)}

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)
    pred_pairs = pred_clusters.unsqueeze(-1) == pred_clusters.unsqueeze(-2)
    gt_pairs = gt_clusters.unsqueeze(-1) == gt_clusters.unsqueeze(-2)
    unmatched_pairs = torch.logical_xor(pred_pairs, gt_pairs).to(dtype=torch.float)
    rand_index = 1 - torch.sum(unmatched_pairs, dim=(-1,-2)) / (num_instances * (num_instances - 1))
    return rand_index


def generate_roc_curve(genuine_scores: Tensor, impostor_scores: Tensor, save_path: str = None):
    """Generate ROC curve for genuine vs impostor matching evaluation.

    This utility computes false positive rates (FPR) and true positive rates
    (TPR) for a series of thresholds given two groups of scores:
    ``genuine_scores`` produced by matching images of the same identity and
    ``impostor_scores`` produced by matching different identities.

    Parameters
    ----------
    genuine_scores : Tensor
        Similarity scores for genuine pairs. Higher scores should indicate
        a better match.
    impostor_scores : Tensor
        Similarity scores for impostor pairs.
    save_path : str, optional
        If provided, the ROC curve will be plotted and saved to this path.

    Returns
    -------
    tuple
        ``(fpr, tpr, roc_auc)`` where ``fpr`` and ``tpr`` are numpy arrays of
        false positive and true positive rates respectively, and ``roc_auc`` is
        the area under the ROC curve.
    """

    import numpy as np
    from sklearn.metrics import roc_curve, auc

    # Convert tensors to numpy arrays on CPU
    genuine_np = genuine_scores.detach().cpu().numpy().ravel()
    impostor_np = impostor_scores.detach().cpu().numpy().ravel()

    scores = np.concatenate([genuine_np, impostor_np])
    labels = np.concatenate([
        np.ones_like(genuine_np, dtype=np.int32),
        np.zeros_like(impostor_np, dtype=np.int32),
    ])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (area = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    return fpr, tpr, roc_auc
