import logging
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from src.model.afau import Encoder
from src.model.affinity_layer import InnerProductWithWeightsAffinity
from src.model.feature_extractor import ResNet34_base as CNN
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
FEATURE_CHANNEL_NODE = 256  # ResNet34 layer3 channels
FEATURE_CHANNEL_EDGE = 512  # ResNet34 layer4 channels
NODE_FEATURE_DIM = FEATURE_CHANNEL_NODE + FEATURE_CHANNEL_EDGE  # 768
GLOBAL_FEATURE_DIM = FEATURE_CHANNEL_EDGE  # 512
GLOBAL_STATE_DIM = GLOBAL_FEATURE_DIM * 2  # 1024

FIRST_ORDER = True
POSITIVE_EDGES = True
GNN_LAYER = 3
SK_TAU= 0.005
# SK_TAU = 0.01
SK_EMB = 3
GNN_FEAT = [16, 16, 16]
EDGE_EMB = True
BATCH_SIZE = 4

UNIV_SIZE = 600
SK_ITER_NUM = 20
SK_EPSILON = 1e-10
K_FACTOR = 50.0
DUSTBIN_LOSS_WEIGHT = 1.0
RESCALE = (256, 256)
CROPSIZE = (224, 224)



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
        # print(k_match)
        # If a pair is an imposter
        if no_pair_mask is not None:
            # k_match = torch.clamp(k_match, min=40)
            # print(k_match)
            k_match = torch.where(no_pair_mask, torch.zeros_like(k_match), k_match)
    else:
        # If in eval, use prediction count
        k_match = k_pred_count
        k_match = torch.round(k_match)

    # k_match = torch.clamp(k_match, min=0.0)
    k_match = torch.minimum(k_match, min_points)
    return k_match


def _dustbin_supervision_loss(pred, gt, nrows, ncols):
    loss = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)
    eps = 1e-6
    # For each prediction in the batch
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


def _dustbin_soft_k_from_transport(transport_with_dustbin, nrows, ncols):
    """Estimate a soft match count K from OT dustbin mass.

    Uses the active dustbin row/col indices (n1, n2) for each sample 
    """
    batch_size = int(transport_with_dustbin.shape[0])
    k_vals = []
    k_rows_vals = []
    k_cols_vals = []
    for b in range(batch_size):
        n1 = int(nrows[b].item())
        n2 = int(ncols[b].item())
        max_k = float(min(max(n1, 0), max(n2, 0)))
        if n1 <= 0 or n2 <= 0:
            zero = transport_with_dustbin.new_tensor(0.0)
            k_vals.append(zero)
            k_rows_vals.append(zero)
            k_cols_vals.append(zero)
            continue

        # Active dustbin column sits at index n2 and active dustbin row at n1.
        row_dustbin_mass = transport_with_dustbin[b, :n1, n2].sum()
        col_dustbin_mass = transport_with_dustbin[b, n1, :n2].sum()

        n1_t = transport_with_dustbin.new_tensor(float(n1))
        n2_t = transport_with_dustbin.new_tensor(float(n2))
        k_hat_rows = n1_t - row_dustbin_mass
        k_hat_cols = n2_t - col_dustbin_mass
        k_hat = 0.5 * (k_hat_rows + k_hat_cols)
        k_hat = torch.clamp(k_hat, min=0.0, max=max_k)
        k_hat_rows = torch.clamp(k_hat_rows, min=0.0, max=max_k)
        k_hat_cols = torch.clamp(k_hat_cols, min=0.0, max=max_k)

        k_vals.append(torch.nan_to_num(k_hat, nan=0.0, posinf=max_k, neginf=0.0))
        k_rows_vals.append(torch.nan_to_num(k_hat_rows, nan=0.0, posinf=max_k, neginf=0.0))
        k_cols_vals.append(torch.nan_to_num(k_hat_cols, nan=0.0, posinf=max_k, neginf=0.0))

    if not k_vals:
        empty = transport_with_dustbin.new_zeros((0,), dtype=torch.float32)
        return empty, empty, empty
    return torch.stack(k_vals), torch.stack(k_rows_vals), torch.stack(k_cols_vals)


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
        dustbin_k_mse_weight: float = 1.0,
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

        self.rescale = RESCALE
        self.cropsize = CROPSIZE
        self.univ_size = UNIV_SIZE
        self.k_factor = K_FACTOR
        self.dustbin_loss_weight = float(dustbin_loss_weight)
        self.dustbin_k_mse_weight = float(dustbin_k_mse_weight)

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
        stage_raw = data_dict.get("stage_id", -1)
        if isinstance(stage_raw, torch.Tensor):
            stage_id = int(stage_raw.reshape(-1)[0].item()) if stage_raw.numel() > 0 else -1
        elif isinstance(stage_raw, (list, tuple)):
            stage_id = int(stage_raw[0]) if len(stage_raw) > 0 else -1
        else:
            try:
                stage_id = int(stage_raw)
            except (TypeError, ValueError):
                stage_id = -1
        stage1_mode = stage_id == 1
        # if any of the k parameters require gradients, set k_trainable
        k_trainable = any(
            bool(p.requires_grad)
            for p in itertools.chain(
                self.encoder_k.parameters(),
                self.final_row.parameters(),
                self.final_col.parameters(),
            )
        )
        # dustbin is trainable, set dustbin trainable
        dustbin_trainable = bool(getattr(self.bin_score, "requires_grad", False))
        # determine if only k or dustbin is trainable
        only_k = bool(self.training) and k_trainable and (not dustbin_trainable)
        only_dustbin = bool(self.training) and dustbin_trainable and (not k_trainable)

        # Load annotations from dataloader
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

            # Load node, edge and global feature maps 
            node_maps = self.node_layers(image)

            edge_maps = self.edge_layers(node_maps)

            global_feature = self.final_layers(edge_maps).reshape((node_maps.shape[0], -1))
            global_features.append(global_feature)

            node_maps = normalize_over_channels(node_maps)
            edge_maps = normalize_over_channels(edge_maps)
            
            # interpolate with points
            node_desc = concat_features(feature_align(node_maps, point, num_p, self.cropsize), num_p)
            edge_desc = concat_features(feature_align(edge_maps, point, num_p, self.cropsize), num_p)

            node_features = torch.cat((node_desc, edge_desc), dim=1)
            
            # Add positional embedding
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

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)

            if FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(Kp.shape[0], Kp.shape[1] * Kp.shape[2], 1, device=Kp.device)
            kgh_sparse = data_dict["KGHs_sparse"]

            qap_emb = []
            sparse_size = Kp.shape[1] * Kp.shape[2]
            for b in range(Kp.shape[0]):
                if num_graphs == 2:
                    kro_G, kro_H = kgh_sparse[b]
                else:
                    key = "{},{}".format(idx1, idx2)
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

            matcher_emb = torch.stack(pad_tensor(qap_emb), dim=0)
            logits = self.classifier(matcher_emb)
            match_scores = logits.view(logits.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)


        # Doubly stochastic matrix for K head (AFAU path)
        ss_base = self.sinkhorn(match_scores, n_points[idx1], n_points[idx2], dummy_row=True)
        ds_mat_matcher = torch.nan_to_num(ss_base, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size, max_n1_transport, max_n2_transport = match_scores.shape
        nrows = n_points[idx1].to(match_scores.device).view(-1)
        ncols = n_points[idx2].to(match_scores.device).view(-1)

        min_point_tensor = torch.minimum(nrows, ncols).to(dtype=torch.float32)
        # if self.training and data_dicKpt.get("label") is not None:
        #     labels = data_dict["label"].to(match_scores.device).view(-1).long()
        #     imposter_mask = labels == 0
        #     if imposter_mask.any():
        #         min_point_tensor = torch.where(imposter_mask, torch.zeros_like(min_point_tensor), min_point_tensor)
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
            ks = torch.sigmoid(k_logits)
            # ks = torch.relu(k_logits)
        else:
            ks = gt_ks / min_point_tensor
            # ks = gt_ks

        # ----- K Match Count Selection -----
        predicted_match_count = torch.nan_to_num(
            ks.view(-1) * min_point_tensor,
            # ks.view(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        selected_match_count = _compute_k_match_count(
            predicted_match_count,
            gt_ks,
            min_point_tensor,
            training=bool(self.training),
            train_use_pred_k=bool(self.train_use_pred_k),
            no_pair_mask=no_pair_mask if self.training else None,
        )
        if stage1_mode:
            # Stage 1 forces GT-k as the selected match count.
            selected_match_count = torch.clamp(gt_ks, min=0.0)
            selected_match_count = torch.minimum(selected_match_count, min_point_tensor)

        # ----- Transport Matrices (Dustbin + Real Block) -----
        # transport_with_dustbin keeps the extra row/col; transport_real_block excludes them.
        transport_with_dustbin = match_scores.new_zeros(batch_size, max_n1_transport + 1, max_n2_transport + 1)
        transport_real_block = match_scores.new_zeros(batch_size, max_n1_transport, max_n2_transport)
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
            sample_logits_real_block = match_scores[b, :n1, :n2]
            log_transport = log_optimal_transport(sample_logits_real_block / self.tau, self.bin_score, SK_ITER_NUM)
            sample_transport_with_dustbin = torch.exp(log_transport)
            sample_transport_with_dustbin = torch.nan_to_num(
                sample_transport_with_dustbin,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            transport_with_dustbin[b, : n1 + 1, : n2 + 1] = sample_transport_with_dustbin
            if n1 > 0 and n2 > 0:
                transport_real_block[b, :n1, :n2] = sample_transport_with_dustbin[:n1, :n2]
                dustbin_col = transport_with_dustbin[b, :n1, -1]   # (n1,)
                dustbin_row = transport_with_dustbin[b, -1, :n2]   # (n2,)

                mask = (
                    transport_with_dustbin[b, :n1, :n2] < dustbin_col[:, None]
                ) & (
                    transport_with_dustbin[b, :n1, :n2] < dustbin_row[None, :]
                )
                transport_real_block[b, :n1, :n2][mask] = 0.0

        dustbin_k_pred_count, dustbin_k_rows_count, dustbin_k_cols_count = _dustbin_soft_k_from_transport(
            transport_with_dustbin,
            nrows,
            ncols,
        )
        if dustbin_k_pred_count.numel() > 0:
            dustbin_k_balance_err = torch.mean(torch.abs(dustbin_k_rows_count - dustbin_k_cols_count))
        else:
            dustbin_k_balance_err = match_scores.new_tensor(0.0)

        # ----- Top-k Refinement Path -----
        if (bool(self.regression) and (not only_dustbin)) or stage1_mode:
            topk_match_count = selected_match_count.view(-1)
            _, topk_selection_prob = soft_topk(
                ss_base,
                topk_match_count,
                SK_ITER_NUM,
                self.tau,
                n_points[idx1],
                n_points[idx2],
                return_prob=True,
            )
            topk_refined_transport = topk_selection_prob 
            topk_refined_transport = torch.nan_to_num(topk_refined_transport, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            topk_refined_transport = transport_real_block.clone()
            topk_refined_transport = torch.nan_to_num(topk_refined_transport, nan=0.0, posinf=0.0, neginf=0.0)



        # ----- DS Matrix Assembly -----  
        ds_mat = transport_with_dustbin.clone()
        ds_mat[:, : topk_refined_transport.shape[1], : topk_refined_transport.shape[2]] = topk_refined_transport

        # ----- Top-k Permutation Proposal -----
        perm_topk_match_count = selected_match_count
        hungarian_seed_perm = hungarian(topk_refined_transport, n_points[idx1], n_points[idx2])
        top_indices = torch.argsort(
            hungarian_seed_perm.mul(topk_refined_transport).reshape(hungarian_seed_perm.shape[0], -1),
            descending=True,
            dim=-1,
        )
        topk_permutation = torch.zeros_like(topk_refined_transport)
        topk_permutation = greedy_perm(topk_permutation, top_indices, perm_topk_match_count)

        
        # if not self.training:
        #     mask = ss_base < 0.3
        #     topk_permutation[mask] = 0

        # ----- Dustbin Rejection/Gating Proposal -----
        dustbin_gated_permutation = torch.zeros_like(transport_real_block)
        reject_rows = []
        reject_cols = []
        dustbin_reject_enable = bool(getattr(self, "dustbin_reject_enable", False))
        margin = transport_with_dustbin.new_tensor(float(getattr(self, "dustbin_reject_margin", 0.0)))
        for b in range(transport_real_block.shape[0]):
            n1 = int(n_points[idx1][b].item())
            n2 = int(n_points[idx2][b].item())
            reject_row_mask = transport_with_dustbin.new_zeros((n1,), dtype=torch.bool)
            reject_col_mask = transport_with_dustbin.new_zeros((n2,), dtype=torch.bool)

            if dustbin_reject_enable and n1 > 0 and n2 > 0:
                sample_transport_full = transport_with_dustbin[b]
                sample_transport_real = transport_real_block[b, :n1, :n2]
                row_best_real_prob = sample_transport_real.max(dim=1).values
                col_best_real_prob = sample_transport_real.max(dim=0).values
                row_dustbin_prob = sample_transport_full[:n1, n2]
                col_dustbin_prob = sample_transport_full[n1, :n2]
                reject_row_mask = row_dustbin_prob >= (row_best_real_prob + margin)
                reject_col_mask = col_dustbin_prob >= (col_best_real_prob + margin)

            kept_row_indices = (~reject_row_mask).nonzero(as_tuple=False).view(-1)
            kept_col_indices = (~reject_col_mask).nonzero(as_tuple=False).view(-1)
            if n1 > 0 and n2 > 0 and kept_row_indices.numel() > 0 and kept_col_indices.numel() > 0:
                kept_submatrix = transport_real_block[b, :n1, :n2].index_select(0, kept_row_indices).index_select(
                    1, kept_col_indices
                )
                kept_submatrix_perm = hungarian(kept_submatrix)
                nz = kept_submatrix_perm.nonzero(as_tuple=False)
                if nz.numel() > 0:
                    rr = kept_row_indices[nz[:, 0]]
                    cc = kept_col_indices[nz[:, 1]]
                    dustbin_gated_permutation[b, rr, cc] = 1.0

            reject_rows.append(reject_row_mask)
            reject_cols.append(reject_col_mask)

        # ----- Stage/Mode Overrides and Final Composition -----
        if stage1_mode:
            # Stage 1 uses top-k-only permutation output by neutralizing dustbin gating.
            dustbin_gated_permutation = torch.ones_like(topk_permutation)
            reject_rows = [transport_with_dustbin.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx1]]
            reject_cols = [transport_with_dustbin.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx2]]
        if only_k:
            dustbin_gated_permutation = torch.ones_like(topk_permutation)
            reject_rows = [transport_with_dustbin.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx1]]
            reject_cols = [transport_with_dustbin.new_zeros((int(n.item()),), dtype=torch.bool) for n in n_points[idx2]]
        if only_dustbin:
            topk_permutation = torch.ones_like(dustbin_gated_permutation)

        perm_mat = topk_permutation * dustbin_gated_permutation

        supervised_ks = torch.where(
            min_point_tensor > 0,
            gt_ks / min_point_tensor_safe,
            torch.zeros_like(gt_ks),
        )
        if no_pair_mask is not None:
            supervised_ks = torch.where(no_pair_mask, torch.zeros_like(supervised_ks), supervised_ks)

    
        
        if bool(self.regression) and k_trainable:
            ks_loss = F.mse_loss(ks, supervised_ks) * self.k_factor
            # ks_loss = F.mse_loss(ks, gt_ks) * self.k_factor
            ks_error = F.l1_loss(ks * min_point_tensor, gt_ks)
            # ks_error = F.l1_loss(ks, gt_ks)
            ks_loss = ks_loss + 0.0001 * ks_error
        else:
            ks_loss = ks.new_tensor(0.0)
            ks_error = ks.new_tensor(0.0)

        if dustbin_trainable and data_dict.get("gt_perm_mat") is not None:
            dustbin_k_mse_loss = F.mse_loss(dustbin_k_pred_count, gt_ks)
            dustbin_k_mae = F.l1_loss(dustbin_k_pred_count, gt_ks)
            dustbin_bce_loss = _dustbin_supervision_loss(
                transport_with_dustbin,
                data_dict["gt_perm_mat"],
                n_points[idx1],
                n_points[idx2],
            )
            dustbin_loss = (
                (self.dustbin_k_mse_weight * dustbin_k_mse_loss)
                + (self.dustbin_loss_weight * dustbin_bce_loss)
            )
        else:
            dustbin_k_mse_loss = ks.new_tensor(0.0)
            dustbin_k_mae = ks.new_tensor(0.0)
            dustbin_bce_loss = ks.new_tensor(0.0)
            dustbin_loss = ks.new_tensor(0.0)

        # print(ks_error * 0.0001)

        data_dict.update(
            {   
                "ds_mat_topk": topk_refined_transport,
                "ds_mat_dustbin": transport_real_block,
                "ds_mat_matcher": ds_mat_matcher,
                "ds_mat": ds_mat,
                "perm_mat": perm_mat,
                "perm_mat_topk": topk_permutation,
                "perm_mat_dustbin": dustbin_gated_permutation,
                "ks_loss": ks_loss,
                "ks_error": ks_error,
                "dustbin_loss": dustbin_loss,
                "dustbin_k_mse_loss": dustbin_k_mse_loss,
                "dustbin_k_mae": dustbin_k_mae,
                "dustbin_bce_loss": dustbin_bce_loss,
                "dustbin_k_balance_err": dustbin_k_balance_err,
                "gt_ks": gt_ks,
                "k_pred_count": predicted_match_count,
                "k_match_count": selected_match_count,
                "k_prob": ks,
                "dustbin_k_pred_count": dustbin_k_pred_count,
                "rejected_rows": reject_rows,
                "rejected_cols": reject_cols,
                "ns": [n_points[idx1] + 1, n_points[idx2] + 1],
                "has_dustbin": True,
            }
        )

        return data_dict
