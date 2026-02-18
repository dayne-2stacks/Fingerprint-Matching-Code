import logging
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from src.model.afau import Encoder
from src.model.affinity_layer import InnerProductWithWeightsAffinity
from src.model.feature_extractor import ResNet18_final as CNN
from src.model.gnn import PYGNNLayer
from src.model.sinkhorn import Sinkhorn
from src.model.soft_topk import greedy_perm, soft_topk
from src.model.spline_conv import SiameseNodeFeaturesToEdgeFeatures, SiameseSConvOnNodes
from utils.factorize_graph_matching import construct_sparse_aff_mat
from utils.feature_align import feature_align
from utils.hungarian import hungarian
from utils.pad_tensor import pad_tensor

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='fp.log',
#     # Remove `encoding='utf-8'` if you have Python < 3.9 or if it causes issues
#     # encoding='utf-8',
#     level=logging.DEBUG
# )


# Params
FEATURE_CHANNEL_NODE = 256  # ResNet18 layer3 channels
FEATURE_CHANNEL_EDGE = 512  # ResNet18 layer4 channels
NODE_FEATURE_DIM = FEATURE_CHANNEL_NODE + FEATURE_CHANNEL_EDGE  # 768
GLOBAL_FEATURE_DIM = FEATURE_CHANNEL_EDGE  # 512
GLOBAL_STATE_DIM = GLOBAL_FEATURE_DIM * 2  # 1024

FIRST_ORDER = True
POSITIVE_EDGES = True
GNN_LAYER = 3
# SK_TAU= 0.005
SK_TAU = 0.01
SK_EMB = 1
GNN_FEAT = [16, 16, 16]
GNN_LAYER = 3
EDGE_EMB = False
BATCH_SIZE = 8

UNIV_SIZE = 450
SK_ITER_NUM = 10
SK_EPSILON = 1e-10
K_FACTOR = 50.0
DUSTBIN_LOSS_WEIGHT = 1.0

def _transport_entropy(block):
    if block.numel() <= 1:
        return block.new_tensor(0.0)
    p = block.reshape(-1).clamp(min=1e-12)
    ent = -(p * torch.log(p)).sum()
    norm = torch.log(block.new_tensor(float(p.numel())))
    return ent / norm.clamp(min=1e-12)


def _topk_confidence_gap(block):
    if block.numel() == 0:
        return block.new_tensor(0.0)
    n2 = block.shape[1]
    if n2 == 1:
        return block[:, 0].mean()
    top2 = torch.topk(block, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).mean()


def _compute_pred_match_ratio(k_pred_count, min_point_tensor_safe):
    return torch.where(
        min_point_tensor_safe > 0,
        k_pred_count / min_point_tensor_safe,
        torch.zeros_like(k_pred_count),
    ).clamp(0.0, 1.0)


def _compute_k_match_count(
    k_pred_count,
    gt_ks,
    min_points,
    *,
    training,
    train_use_pred_k,
    no_pair_mask,
):
    if training:
        k_match = k_pred_count if bool(train_use_pred_k) else gt_ks
        if no_pair_mask is not None:
            k_match = torch.where(no_pair_mask, torch.zeros_like(k_match), k_match)
    else:
        k_match = k_pred_count
        
        k_match = torch.round(k_match)

    k_match = torch.clamp(k_match, min=0.0)
    k_match = torch.minimum(k_match, min_points)
    return k_match


def _dustbin_supervision_loss(pred, gt, nrows, ncols):
    loss = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)
    eps = 1e-6
    for b in range(pred.shape[0]):
        n1 = int(nrows[b].item())
        n2 = int(ncols[b].item())
        if n1 > 0:
            pred_col = pred[b, :n1, n2].clamp(min=eps, max=1.0 - eps)
            gt_col = gt[b, :n1, n2].to(pred.dtype)
            loss += F.binary_cross_entropy(pred_col, gt_col, reduction="sum")
            count += n1
        if n2 > 0:
            pred_row = pred[b, n1, :n2].clamp(min=eps, max=1.0 - eps)
            gt_row = gt[b, n1, :n2].to(pred.dtype)
            loss += F.binary_cross_entropy(pred_row, gt_row, reduction="sum")
            count += n2
    if count > 0:
        loss = loss / count
    return loss


# Return as iterable combinations
def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-12)
    return x / channel_norms


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image shape."""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def _nan_stats(tag, x):
    if x is None:
        logger.info("%s: None", tag)
        return
    logger.info(
        "%s: nan=%s inf=%s min=%s max=%s",
        tag,
        torch.isnan(x).any().item(),
        torch.isinf(x).any().item(),
        x.min().item(),
        x.max().item(),
    )


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


def log_sinkhorn_iterations(z, log_mu, log_nu, iters):
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(1), dim=0)
    return z + u.unsqueeze(1) + v.unsqueeze(0)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability (2D scores)."""
    m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(m, 1)
    bins1 = alpha.expand(1, n)
    alpha = alpha.view(1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 0)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])

    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    z = z - norm  # multiply probabilities by M+N
    return z


# CNN is the VGG16 feature extractor with final fully connected layers

# Contains methods {
#     node_layers - features for nodes (pores)
#     edge_layers - features for edges
#     final_layers - global features of larger fingerprint
# }
class Net(CNN):
    def __init__(
        self,
        regression: bool = False,
        mean_k: bool = True,
        dustbin_loss_weight: float = 0.5,
        dustbin_reject_enable: bool = False,
        dustbin_reject_margin: float = 0.0,
        train_use_pred_k: bool = False,
    ):
        super(Net, self).__init__()  # initialize the VGG16 model

        # --- Spline-Conv path ------------------------------------------------
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=NODE_FEATURE_DIM)

        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )

        # --- Affinity layers -------------------------------------------------
        self.global_state_dim = GLOBAL_STATE_DIM  # 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.message_pass_node_features.num_node_features,
        )

        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features,
        )

        self.tau = SK_TAU

        # Initialize my GNN Layer
        self.gnn_layer = GNN_LAYER
        for i in range(self.gnn_layer):
            tau = self.tau
            if i == 0:
                gnn_layer = PYGNNLayer(
                    1,
                    1,
                    GNN_FEAT[i] + SK_EMB,
                    GNN_FEAT[i],
                    sk_channel=SK_EMB,
                    sk_tau=tau,
                    edge_emb=EDGE_EMB,
                )
            else:
                gnn_layer = PYGNNLayer(
                    GNN_FEAT[i - 1] + SK_EMB,
                    GNN_FEAT[i - 1],
                    GNN_FEAT[i] + SK_EMB,
                    GNN_FEAT[i],
                    sk_channel=SK_EMB,
                    sk_tau=tau,
                    edge_emb=EDGE_EMB,
                )
            self.add_module("gnn_layer_{}".format(i), gnn_layer)

        self.rescale = (320, 240)
        self.univ_size = UNIV_SIZE
        self.k_factor = K_FACTOR
        self.dustbin_loss_weight = float(dustbin_loss_weight)

        # Classify fingerprint
        self.classifier = nn.Linear(GNN_FEAT[-1] + SK_EMB, 1)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, NODE_FEATURE_DIM),
        )

        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, tau=self.tau, epsilon=SK_EPSILON)
        self.regression = bool(regression)
        self.mean_k = bool(mean_k)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self.train_use_pred_k = bool(train_use_pred_k)
        self.dustbin_reject_enable = bool(dustbin_reject_enable)
        self.dustbin_reject_margin = float(max(float(dustbin_reject_margin), 0.0))

        
        self.k_params_id = []
        # Only implementing AFAU
        self.encoder_k = Encoder()
        self.k_params_id += [id(item) for item in self.encoder_k.parameters()]
        self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
        self.final_row = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.final_col = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.k_params_id += [id(item) for item in self.final_row.parameters()]
        self.k_params_id += [id(item) for item in self.final_col.parameters()]

        self.k_params = [
            {"params": self.encoder_k.parameters()},
            {"params": self.final_row.parameters()},
            {"params": self.final_col.parameters()},
        ]

    def forward(self, data_dict, regression=True):
        del regression  # kept for interface compatibility
        k_trainable = any(
            bool(p.requires_grad)
            for p in itertools.chain(
                self.encoder_k.parameters(),
                self.final_row.parameters(),
                self.final_col.parameters(),
            )
        )
        dustbin_trainable = bool(getattr(self.bin_score, "requires_grad", False))
        only_k = bool(self.training) and k_trainable and (not dustbin_trainable)
        only_dustbin = bool(self.training) and dustbin_trainable and (not k_trainable)

        images = data_dict["images"]  # Loaded from custom dataset
        points = data_dict["Ps"]  # Pore locations
        n_points = data_dict["ns"]  # number of pores
        graphs = data_dict["pyg_graphs"]  # Generated by GMDataset
        num_graphs = len(images)  # number of fingerprints
        log_nans = not getattr(self, "_nan_debug_done", False)

        global_features = []
        processed_graphs = []
        for image, point, num_p, graph in zip(images, points, n_points, graphs):
            if image.dim() == 3:
                image = image.unsqueeze(0)

            node_maps = self.node_layers(image)

            edge_maps = self.edge_layers(node_maps)

            global_feature = self.final_layers(edge_maps).reshape((node_maps.shape[0], -1))
            global_features.append(global_feature)

            node_maps = normalize_over_channels(node_maps)
            edge_maps = normalize_over_channels(edge_maps)

            node_desc = concat_features(feature_align(node_maps, point, num_p, self.rescale), num_p)
            edge_desc = concat_features(feature_align(edge_maps, point, num_p, self.rescale), num_p)

            node_features = torch.cat((node_desc, edge_desc), dim=1)

            pos = normalize_keypoints(point, image.shape)
            pos_emb = self.pos_mlp(pos).permute(0, 2, 1)
            pos_emb = concat_features(pos_emb, num_p)
            node_features = node_features + pos_emb
            graph.x = node_features

            graph = self.message_pass_node_features(graph)

            edge_graph = self.build_edge_features_from_node_features(graph)
            processed_graphs.append(edge_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1)
            for global_src, global_tgt in lexico_iter(global_features)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        pair_indices = list(lexico_iter(range(num_graphs)))
        if len(pair_indices) != 1:
            raise ValueError(
                f"Net.forward currently supports 2GM pairs only. Expected 1 pair, got {len(pair_indices)}"
            )

        idx1, idx2 = pair_indices[0]
        unary_affs = unary_affs_list[0]
        quadratic_affs = quadratic_affs_list[0]

        if "KGHs_sparse" not in data_dict:
            raise KeyError("data_dict is missing required key 'KGHs_sparse'")
        Kp = torch.stack(pad_tensor(unary_affs), dim=0)
        if Kp.ndim != 3:
            raise ValueError(f"Expected Kp to be 3D, got shape {tuple(Kp.shape)}")
        if FIRST_ORDER:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        else:
            emb = torch.ones(Kp.shape[0], Kp.shape[1] * Kp.shape[2], 1, device=Kp.device)
        kgh_sparse = data_dict["KGHs_sparse"]
        if num_graphs == 2 and len(kgh_sparse) != Kp.shape[0]:
            raise ValueError(f"KGHs_sparse batch mismatch: expected {Kp.shape[0]}, got {len(kgh_sparse)}")
        qap_emb = []
        sparse_size = Kp.shape[1] * Kp.shape[2]
        for b in range(Kp.shape[0]):
            if num_graphs == 2:
                kro_G, kro_H = kgh_sparse[b]
            else:
                key = "{},{}".format(idx1, idx2)
                if key not in kgh_sparse:
                    raise KeyError(f"Missing sparse KGH entry for pair key '{key}'")
                kro_G, kro_H = kgh_sparse[key]
            K_value, row_idx, col_idx = construct_sparse_aff_mat(
                quadratic_affs[b], unary_affs[b], kro_G, kro_H
            )
            common_len = min(row_idx.numel(), col_idx.numel(), K_value.numel())
            row = row_idx[:common_len].long()
            col = col_idx[:common_len].long()
            val = K_value[:common_len]
            adj = SparseTensor(
                row=row,
                col=col,
                value=val,
                sparse_sizes=(sparse_size, sparse_size),
            )
            tmp_emb = emb[b].unsqueeze(0)
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, "gnn_layer_{}".format(i))
                tmp_emb = gnn_layer(adj, tmp_emb, n_points[idx1], n_points[idx2], b)
            qap_emb.append(tmp_emb.squeeze(0))
        if len(qap_emb) != Kp.shape[0]:
            raise RuntimeError(f"Unexpected qap_emb length {len(qap_emb)} (expected {Kp.shape[0]})")
        matcher_emb = torch.stack(pad_tensor(qap_emb), dim=0)
        logits = self.classifier(matcher_emb)
        match_scores = logits.view(logits.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)
        max_n1, max_n2 = Kp.shape[1], Kp.shape[2]

        # Doubly stochastic matrix for K head (AFAU path)
        ss_base = self.sinkhorn(match_scores, n_points[idx1], n_points[idx2], dummy_row=True)

        if match_scores.ndim != 3:
            raise ValueError(f"Expected scores to be 3D (B,N,M), got {tuple(match_scores.shape)}")
        batch_size, max_n1_transport, max_n2_transport = match_scores.shape
        nrows = n_points[idx1].to(match_scores.device).view(-1)
        ncols = n_points[idx2].to(match_scores.device).view(-1)
        if nrows.numel() != batch_size or ncols.numel() != batch_size:
            raise ValueError(
                "n_points batch size mismatch: "
                f"scores={batch_size}, nrows={nrows.numel()}, ncols={ncols.numel()}"
            )
        min_point_tensor = torch.minimum(nrows, ncols).to(dtype=torch.float32)
        min_point_tensor_safe = torch.clamp(min_point_tensor, min=1.0)
        gt_perm_mat = data_dict.get("gt_perm_mat")
        if gt_perm_mat is None:
            gt_ks = torch.zeros(batch_size, dtype=torch.float32, device=match_scores.device)
        else:
            if gt_perm_mat.ndim != 3:
                raise ValueError(f"Expected gt_perm_mat to be 3D, got shape {tuple(gt_perm_mat.shape)}")
            if gt_perm_mat.shape[0] != batch_size:
                raise ValueError(
                    f"gt_perm_mat batch mismatch: expected {batch_size}, got {gt_perm_mat.shape[0]}"
                )
            max_n1_local = int(nrows.max().item()) if batch_size > 0 else 0
            max_n2_local = int(ncols.max().item()) if batch_size > 0 else 0
            if gt_perm_mat.shape[1] < max_n1_local or gt_perm_mat.shape[2] < max_n2_local:
                raise ValueError(
                    "gt_perm_mat is smaller than requested n_points real block: "
                    f"gt_perm_mat={tuple(gt_perm_mat.shape)}, max_n1={max_n1_local}, max_n2={max_n2_local}"
                )
            gt_ks = torch.tensor(
                [
                    torch.sum(
                        gt_perm_mat[b, : int(nrows[b].item()), : int(ncols[b].item())]
                    )
                    for b in range(batch_size)
                ],
                dtype=torch.float32,
                device=match_scores.device,
            )
        no_pair_mask = None
        if self.training:
            if "label" in data_dict:
                labels = data_dict["label"].to(match_scores.device).view(-1).long()
                imposter_mask = labels == 0
                if imposter_mask.any():
                    gt_ks = gt_ks.clone()
                    gt_ks[imposter_mask] = 0.0
                no_pair_mask = imposter_mask | (gt_ks == 0)
            else:
                no_pair_mask = gt_ks == 0

        if self.regression:
            dummy_row = self.univ_size - ss_base.shape[1]
            dummy_col = self.univ_size - ss_base.shape[2]
            if dummy_row < 0 or dummy_col < 0:
                raise ValueError(
                    f"AFAU univ_size={self.univ_size} is smaller than sinkhorn score size "
                    f"({ss_base.shape[1]}, {ss_base.shape[2]})"
                )
            max_n1_k = int(torch.max(nrows).item()) if nrows.numel() > 0 else 0
            max_n2_k = int(torch.max(ncols).item()) if ncols.numel() > 0 else 0
            init_row_emb = torch.zeros(
                (batch_size, max_n1_k, self.univ_size),
                dtype=torch.float32,
                device=ss_base.device,
            )
            init_col_emb = torch.zeros(
                (batch_size, max_n2_k, self.univ_size),
                dtype=torch.float32,
                device=ss_base.device,
            )
            for b in range(batch_size):
                n2_b = int(ncols[b].item())
                if n2_b <= 0:
                    continue
                index = torch.arange(n2_b, dtype=torch.long, device=ss_base.device).unsqueeze(1)
                init_col_emb_one = torch.zeros(
                    max_n2_k,
                    self.univ_size,
                    dtype=torch.float32,
                    device=ss_base.device,
                ).scatter_(1, index, 1)
                init_col_emb[b] = init_col_emb_one
            out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss_base)
            out_emb_row = F.pad(out_emb_row, (0, 0, 0, dummy_row), value=float("-inf")).permute(0, 2, 1)
            out_emb_col = F.pad(out_emb_col, (0, 0, 0, dummy_col), value=float("-inf")).permute(0, 2, 1)
            global_row_emb = self.maxpool(out_emb_row).squeeze(-1)
            global_col_emb = self.maxpool(out_emb_col).squeeze(-1)
            k_row_logit = self.final_row(global_row_emb).squeeze(-1)
            k_col_logit = self.final_col(global_col_emb).squeeze(-1)
            if self.mean_k:
                k_logits = (k_row_logit + k_col_logit) / 2
            else:
                k_logits = k_row_logit
            ks = torch.relu(k_logits)/ (1+torch.relu(k_logits))
        else:
            k_logits = torch.zeros_like(min_point_tensor)
            ks = torch.zeros_like(min_point_tensor)

        k_pred_count = torch.nan_to_num(
            ks.view(-1) * min_point_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        k_match = _compute_k_match_count(
            k_pred_count,
            gt_ks,
            min_point_tensor,
            training=bool(self.training),
            train_use_pred_k=bool(self.train_use_pred_k),
            no_pair_mask=no_pair_mask if self.training else None,
        )

        full_transport = match_scores.new_zeros(batch_size, max_n1_transport + 1, max_n2_transport + 1)
        real_transport = match_scores.new_zeros(batch_size, max_n1_transport, max_n2_transport)
        for b in range(batch_size):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 < 0 or n2 < 0:
                raise ValueError(f"Negative graph size detected: n1={n1}, n2={n2}")
            if n1 > max_n1_transport or n2 > max_n2_transport:
                raise ValueError(
                    f"Requested active size exceeds padded score shape: "
                    f"n1={n1}, n2={n2}, max_n1={max_n1_transport}, max_n2={max_n2_transport}"
                )
            if n1 == 0 and n2 == 0:
                continue
            logits_transport = match_scores[b, :n1, :n2]
            log_transport = log_optimal_transport(logits_transport / self.tau, self.bin_score, SK_ITER_NUM)
            transport = torch.exp(log_transport)
            transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)
            full_transport[b, : n1 + 1, : n2 + 1] = transport
            if n1 > 0 and n2 > 0:
                real_transport[b, :n1, :n2] = transport[:n1, :n2]

        
        if bool(self.regression) and (not only_dustbin):
            match_count_for_topk = k_match.view(-1)
            _, topk_prob = soft_topk(
                real_transport,
                match_count_for_topk,
                SK_ITER_NUM,
                self.tau,
                n_points[idx1],
                n_points[idx2],
                return_prob=True,
            )
            refined_real = topk_prob * real_transport
            refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            refined_real = real_transport.clone()
            refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)

        ds_mat_real = real_transport if only_dustbin else refined_real
        ds_mat = full_transport.clone()
        ds_mat[:, : real_transport.shape[1], : real_transport.shape[2]] = ds_mat_real

        match_count_for_perm_topk = k_match if bool(self.regression) else min_point_tensor
        x = hungarian(refined_real, n_points[idx1], n_points[idx2])
        top_indices = torch.argsort(
            x.mul(refined_real).reshape(x.shape[0], -1),
            descending=True,
            dim=-1,
        )
        perm_mat_topk = torch.zeros_like(refined_real)
        perm_mat_topk = greedy_perm(perm_mat_topk, top_indices, match_count_for_perm_topk)

        perm_mat_dustbin = torch.zeros_like(real_transport)
        reject_rows = []
        reject_cols = []
        dustbin_reject_enable = bool(getattr(self, "dustbin_reject_enable", False))
        margin = full_transport.new_tensor(float(getattr(self, "dustbin_reject_margin", 0.0)))
        for b in range(real_transport.shape[0]):
            n1 = int(n_points[idx1][b].item())
            n2 = int(n_points[idx2][b].item())
            row_reject = full_transport.new_zeros((n1,), dtype=torch.bool)
            col_reject = full_transport.new_zeros((n2,), dtype=torch.bool)

            if dustbin_reject_enable and n1 > 0 and n2 > 0:
                p_full = full_transport[b]
                p_real = real_transport[b, :n1, :n2]
                row_best = p_real.max(dim=1).values
                col_best = p_real.max(dim=0).values
                dust_row = p_full[:n1, n2]
                dust_col = p_full[n1, :n2]
                row_reject = dust_row >= (row_best + margin)
                col_reject = dust_col >= (col_best + margin)

            keep_rows = (~row_reject).nonzero(as_tuple=False).view(-1)
            keep_cols = (~col_reject).nonzero(as_tuple=False).view(-1)
            if n1 > 0 and n2 > 0 and keep_rows.numel() > 0 and keep_cols.numel() > 0:
                sub = real_transport[b, :n1, :n2].index_select(0, keep_rows).index_select(1, keep_cols)
                sub_perm = hungarian(sub)
                nz = sub_perm.nonzero(as_tuple=False)
                if nz.numel() > 0:
                    rr = keep_rows[nz[:, 0]]
                    cc = keep_cols[nz[:, 1]]
                    perm_mat_dustbin[b, rr, cc] = 1.0

            reject_rows.append(row_reject)
            reject_cols.append(col_reject)

        if only_k:
            perm_mat_dustbin = torch.ones_like(perm_mat_topk)
            reject_rows = [full_transport.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx1]]
            reject_cols = [full_transport.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx2]]
        if only_dustbin:
            perm_mat_topk = torch.ones_like(perm_mat_dustbin)

        perm_mat = perm_mat_topk * perm_mat_dustbin

        supervised_ks = torch.where(
            min_point_tensor > 0,
            gt_ks / min_point_tensor_safe,
            torch.zeros_like(gt_ks),
        )
        if no_pair_mask is not None:
            supervised_ks = torch.where(no_pair_mask, torch.zeros_like(supervised_ks), supervised_ks)

        # Authentication features/logits must stay prediction-driven only.
        pred_match_ratio = _compute_pred_match_ratio(k_pred_count, min_point_tensor_safe)
        dustbin_mass_row = []
        dustbin_mass_col = []
        transport_entropy = []
        topk_conf_gap = []
        for b in range(full_transport.shape[0]):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            p_full = full_transport[b]
            p_real = real_transport[b, :n1, :n2] if (n1 > 0 and n2 > 0) else real_transport.new_zeros((0, 0))
            if n1 > 0:
                dustbin_mass_row.append(p_full[:n1, n2].mean())
            else:
                dustbin_mass_row.append(p_full.new_tensor(0.0))
            if n2 > 0:
                dustbin_mass_col.append(p_full[n1, :n2].mean())
            else:
                dustbin_mass_col.append(p_full.new_tensor(0.0))
            if n1 > 0 and n2 > 0:
                transport_entropy.append(_transport_entropy(p_real))
                topk_conf_gap.append(_topk_confidence_gap(p_real))
            else:
                zero = p_full.new_tensor(0.0)
                transport_entropy.append(zero)
                topk_conf_gap.append(zero)
        dustbin_mass_row = torch.stack(dustbin_mass_row, dim=0).to(pred_match_ratio.dtype)
        dustbin_mass_col = torch.stack(dustbin_mass_col, dim=0).to(pred_match_ratio.dtype)
        transport_entropy = torch.stack(transport_entropy, dim=0).to(pred_match_ratio.dtype)
        topk_conf_gap = torch.stack(topk_conf_gap, dim=0).to(pred_match_ratio.dtype)

        zero = ks.new_tensor(0.0)
        if bool(self.regression) and k_trainable:
            ks_loss = F.mse_loss(ks, supervised_ks) * self.k_factor
            ks_error = F.l1_loss(ks * min_point_tensor, gt_ks)
        else:
            ks_loss = zero
            ks_error = zero

        if dustbin_trainable and data_dict.get("gt_perm_mat") is not None:
            dustbin_loss = _dustbin_supervision_loss(
                ds_mat,
                data_dict["gt_perm_mat"],
                n_points[idx1],
                n_points[idx2],
            )
            dustbin_loss = dustbin_loss * self.dustbin_loss_weight
        else:
            dustbin_loss = zero

        dustbin_margin_loss = ds_mat.new_tensor(0.0)

        data_dict.update(
            {
                "ds_mat": ds_mat,
                "perm_mat": perm_mat,
                "perm_mat_topk": perm_mat_topk,
                "perm_mat_dustbin": perm_mat_dustbin,
                "ks_loss": ks_loss,
                "ks_error": ks_error,
                "dustbin_loss": dustbin_loss,
                "gt_ks": gt_ks,
                "k_pred_ratio": pred_match_ratio,
                "k_pred_count": k_pred_count,
                "k_match_count": k_match,
                "k_prob": ks,
                "k_logit": k_logits if k_logits is not None else torch.zeros_like(min_point_tensor),
                "dustbin_margin_loss": dustbin_margin_loss,
                "dustbin_margin_value": ds_mat.new_tensor(float(self.dustbin_reject_margin)),
                "dustbin_margin_loss_weight": 0.0,
                "transport_entropy": transport_entropy,
                "dustbin_mass_row": dustbin_mass_row,
                "dustbin_mass_col": dustbin_mass_col,
                "topk_conf_gap": topk_conf_gap,
                "rejected_rows": reject_rows,
                "rejected_cols": reject_cols,
                "ns": [n_points[idx1] + 1, n_points[idx2] + 1],
                "has_dustbin": True,
            }
        )

        return data_dict
