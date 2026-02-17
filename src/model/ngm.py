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

# Hard-coded training dynamics for learnable thresholds.
K_GATE_TEMP = 0.05
K_GATE_LOSS_WEIGHT = 0.2
K_GATE_LEARN_STAGE_MIN = 3
AUTH_GATE_TEMP = 0.05
AUTH_GATE_LOSS_WEIGHT = 0.2
DUSTBIN_MARGIN_TEMP = 0.05
DUSTBIN_MARGIN_LOSS_WEIGHT = 0.2


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
        regression=False,
        k_reg_weight=0.2,
        k_cls_weight=1.0,
        dustbin_loss_weight=0.5,
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
        self.k_reg_weight = float(k_reg_weight)
        self.k_cls_weight = float(k_cls_weight)
        self.dustbin_loss_weight = float(dustbin_loss_weight)

        # Classify fingerprint
        self.classifier = nn.Linear(GNN_FEAT[-1] + SK_EMB, 1)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, NODE_FEATURE_DIM),
        )

        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, tau=self.tau, epsilon=SK_EPSILON)
        self.regression = regression
        self.mean_k = True
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        # Stage policy values are configured via apply_stage_policy().
        self.k_gate_enable = False
        self.k_gate_thresh = 0.2
        self.k_gate_logit = nn.Parameter(
            torch.tensor(self._safe_logit(self.k_gate_thresh), dtype=torch.float32)
        )
        self.k_match_rounding = "round"
        self.auth_gate_enable = False
        self.auth_gate_thresh = 0.5
        self.auth_gate_logit = nn.Parameter(
            torch.tensor(self._safe_logit(self.auth_gate_thresh), dtype=torch.float32)
        )
        self.dustbin_reject_enable = False
        self.dustbin_reject_margin = 0.0
        self.dustbin_reject_margin_logit = nn.Parameter(
            torch.tensor(self._safe_inverse_softplus(self.dustbin_reject_margin), dtype=torch.float32)
        )
        self.k_pred_ramp = 0.0
        self.train_use_pred_k = False
        self.auth_use_pooled = True
        self.auth_scalar_dim = 6
        self.k_match_detach_stage_threshold = 2
        self.auth_pooled_dim = 16
        self.train_stage = None
        self.match_branch = "topk"
        self._auth_pool_warned = False
        self.auth_pool_proj = nn.Sequential(
            nn.Linear(GNN_FEAT[-1] + SK_EMB, 32),
            nn.ReLU(),
            nn.Linear(32, self.auth_pooled_dim),
            nn.ReLU(),
        )
        auth_head_in_dim = self.auth_scalar_dim + (self.auth_pooled_dim if self.auth_use_pooled else 0)
        # Authentication head over engineered + pooled matching features.
        self.auth_head = nn.Sequential(
            nn.Linear(auth_head_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

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

    def apply_stage_policy(self, policy):
        if not isinstance(policy, dict):
            raise TypeError("policy must be a dictionary")

        self.regression = bool(policy.get("REGRESSION", self.regression))
        self.train_use_pred_k = bool(policy.get("TRAIN_USE_PRED_K", self.train_use_pred_k))
        self.k_reg_weight = float(policy.get("K_REG_WEIGHT", self.k_reg_weight))
        self.k_cls_weight = float(policy.get("K_CLS_WEIGHT", self.k_cls_weight))
        self.dustbin_loss_weight = float(policy.get("DUSTBIN_LOSS_WEIGHT", self.dustbin_loss_weight))

        self.k_gate_enable = bool(policy.get("K_GATE_ENABLE", self.k_gate_enable))
        k_gate_thresh = float(policy.get("K_GATE_THRESH", self.k_gate_thresh))
        self.k_gate_thresh = k_gate_thresh
        with torch.no_grad():
            self.k_gate_logit.copy_(
                torch.tensor(
                    self._safe_logit(k_gate_thresh),
                    dtype=self.k_gate_logit.dtype,
                    device=self.k_gate_logit.device,
                )
            )

        self.auth_gate_enable = bool(policy.get("AUTH_GATE_ENABLE", self.auth_gate_enable))
        auth_gate_thresh = float(policy.get("AUTH_GATE_THRESH", self.auth_gate_thresh))
        self.auth_gate_thresh = auth_gate_thresh
        with torch.no_grad():
            self.auth_gate_logit.copy_(
                torch.tensor(
                    self._safe_logit(auth_gate_thresh),
                    dtype=self.auth_gate_logit.dtype,
                    device=self.auth_gate_logit.device,
                )
            )

        self.k_match_rounding = str(policy.get("K_MATCH_ROUNDING", self.k_match_rounding)).strip().lower()
        self.dustbin_reject_enable = bool(
            policy.get("DUSTBIN_REJECT_ENABLE", self.dustbin_reject_enable)
        )
        self.set_dustbin_reject_margin(
            float(policy.get("DUSTBIN_REJECT_MARGIN", self.dustbin_reject_margin))
        )
        self.k_match_detach_stage_threshold = int(
            policy.get("K_MATCH_DETACH_STAGE_THRESHOLD", self.k_match_detach_stage_threshold)
        )
        match_branch = str(policy.get("MATCH_BRANCH", self.match_branch)).strip().lower()
        if match_branch not in {"shared_matcher", "topk", "dustbin"}:
            raise ValueError(
                f"Unsupported MATCH_BRANCH={match_branch}. "
                "Expected one of {'shared_matcher', 'topk', 'dustbin'}."
            )
        self.match_branch = match_branch

        desired_auth_use_pooled = bool(policy.get("AUTH_USE_POOLED", self.auth_use_pooled))
        if desired_auth_use_pooled != self.auth_use_pooled:
            raise ValueError(
                f"AUTH_USE_POOLED={desired_auth_use_pooled} is incompatible with hardcoded model "
                f"architecture AUTH_USE_POOLED={self.auth_use_pooled}"
            )
        desired_auth_scalar_dim = int(policy.get("AUTH_SCALAR_DIM", self.auth_scalar_dim))
        if desired_auth_scalar_dim != self.auth_scalar_dim:
            raise ValueError(
                f"AUTH_SCALAR_DIM={desired_auth_scalar_dim} is incompatible with hardcoded model "
                f"architecture AUTH_SCALAR_DIM={self.auth_scalar_dim}"
            )

        return self

    @staticmethod
    def _safe_logit(p):
        p = float(p)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return float(torch.log(torch.tensor(p / (1.0 - p))).item())

    @staticmethod
    def _safe_inverse_softplus(x):
        x = max(float(x), 1e-6)
        return float(torch.log(torch.expm1(torch.tensor(x))).item())

    def set_dustbin_reject_margin(self, margin):
        self.dustbin_reject_margin = float(max(float(margin), 0.0))
        with torch.no_grad():
            self.dustbin_reject_margin_logit.copy_(
                torch.tensor(
                    self._safe_inverse_softplus(self.dustbin_reject_margin),
                    dtype=self.dustbin_reject_margin_logit.dtype,
                    device=self.dustbin_reject_margin_logit.device,
                )
            )

    def _current_k_gate_thresh(self):
        return torch.sigmoid(self.k_gate_logit)

    def _current_auth_gate_thresh(self):
        return torch.sigmoid(self.auth_gate_logit)

    def _current_dustbin_reject_margin(self):
        return F.softplus(self.dustbin_reject_margin_logit)

    def _k_gate_learning_enabled(self):
        if not self.training:
            return False
        stage = getattr(self, "train_stage", None)
        try:
            stage_num = int(stage)
        except (TypeError, ValueError):
            return False
        return stage_num >= int(K_GATE_LEARN_STAGE_MIN)

    def _compute_k_gate_loss(self, ks, labels):
        thresh = self._current_k_gate_thresh().to(ks.device, dtype=ks.dtype)
        gate_prob = torch.sigmoid((ks.view(-1) - thresh) / float(K_GATE_TEMP))
        gate_loss = F.binary_cross_entropy(gate_prob, labels.view(-1).float())
        return gate_loss, gate_prob

    def _compute_auth_gate_loss(self, auth_prob, labels):
        thresh = self._current_auth_gate_thresh().to(auth_prob.device, dtype=auth_prob.dtype)
        gate_prob = torch.sigmoid((auth_prob.view(-1) - thresh) / float(AUTH_GATE_TEMP))
        gate_loss = F.binary_cross_entropy(gate_prob, labels.view(-1).float())
        return gate_loss, gate_prob

    def _extract_features(self, images, points, n_points, graphs, log_nans):
        global_features = []
        processed_graphs = []

        # Loop through images, pores, number of pores, graphs
        for image, point, num_p, graph in zip(images, points, n_points, graphs):
            # if a single image is being passed, unsqueeze dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Get node, edge and global features from image (Using ResNet18 layers)
            node_maps = self.node_layers(image)
            if log_nans:
                _nan_stats("node_layers", node_maps)

            edge_maps = self.edge_layers(node_maps)
            if log_nans:
                _nan_stats("edge_layers", edge_maps)

            global_feature = self.final_layers(edge_maps).reshape((node_maps.shape[0], -1))
            global_features.append(global_feature)
            if log_nans:
                _nan_stats("final_layers", global_feature)

            # L2 Norm
            node_maps = normalize_over_channels(node_maps)
            edge_maps = normalize_over_channels(edge_maps)
            if log_nans:
                _nan_stats("norm_nodes", node_maps)
                _nan_stats("norm_edges", edge_maps)

            # arrange features
            node_desc = concat_features(feature_align(node_maps, point, num_p, self.rescale), num_p)
            edge_desc = concat_features(feature_align(edge_maps, point, num_p, self.rescale), num_p)
            if log_nans:
                _nan_stats("feature_align_U", node_desc)
                _nan_stats("feature_align_F", edge_desc)

            node_features = torch.cat((node_desc, edge_desc), dim=1)
            if log_nans:
                _nan_stats("node_features_cat", node_features)

            pos = normalize_keypoints(point, image.shape)
            # Embed positional information and add to node features
            pos_emb = self.pos_mlp(pos).permute(0, 2, 1)
            pos_emb = concat_features(pos_emb, num_p)

            # Update node features with positional embedding
            node_features = node_features + pos_emb
            graph.x = node_features

            # Apply Spline conv network for enhanced feature extraction
            graph = self.message_pass_node_features(graph)
            if log_nans:
                _nan_stats("message_pass_node_features", graph.x)

            edge_graph = self.build_edge_features_from_node_features(graph)
            if log_nans:
                edge_attr = edge_graph[0].edge_attr if isinstance(edge_graph, list) else edge_graph.edge_attr
                _nan_stats("build_edge_features_from_node_features", edge_attr)

            processed_graphs.append(edge_graph)

        return global_features, processed_graphs

    def _build_affinities(self, processed_graphs, global_features, log_nans):
        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1)
            for global_src, global_tgt in lexico_iter(global_features)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]
        if log_nans:
            _nan_stats("global_weights_list[0]", global_weights_list[0] if global_weights_list else None)

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]
        if log_nans and unary_affs_list:
            _nan_stats("unary_affs_list[0]", unary_affs_list[0][0] if unary_affs_list[0] else None)

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]
        if log_nans and quadratic_affs_list:
            _nan_stats("quadratic_affs_list[0]", quadratic_affs_list[0][0] if quadratic_affs_list[0] else None)

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]
        return unary_affs_list, quadratic_affs_list

    def _run_gnn(self, unary_affs, quadratic_affs, pair_index, data_dict, n_points, num_graphs):
        idx1, idx2 = pair_index
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
            raise ValueError(
                f"KGHs_sparse batch mismatch: expected {Kp.shape[0]}, got {len(kgh_sparse)}"
            )

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

            # Ensure index/value tensors have the same length when constructing
            # the sparse affinity matrix. Occasionally the returned row/col
            # indices may not perfectly match the value tensor due to
            # preprocessing irregularities. To prevent runtime errors in
            # ``torch_sparse`` we truncate all tensors to the smallest common
            # length.
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
            raise RuntimeError(
                f"Unexpected qap_emb length {len(qap_emb)} (expected {Kp.shape[0]})"
            )

        emb_out = torch.stack(pad_tensor(qap_emb), dim=0)
        logits = self.classifier(emb_out)
        scores = logits.view(logits.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)
        return scores, emb_out, (Kp.shape[1], Kp.shape[2])

    def _should_detach_for_k(self):
        stage = getattr(self, "train_stage", None)
        if stage is None:
            return False
        try:
            stage_num = int(stage)
        except (TypeError, ValueError):
            return False
        return stage_num < int(self.k_match_detach_stage_threshold)

    def _compute_pair_metadata(self, data_dict, idx1, idx2, scores):
        """Compute pair metadata for matching and supervision.

        Returns:
            min_point_tensor: min(n1, n2) per sample.
            min_point_tensor_safe: clamped min points used for safe division.
            gt_ks: supervision-only GT correspondence counts (for losses/logging).
            no_pair_mask: training-only mask for enforcing zero matches.
        """
        # Scores must be in the form (B,N,M)
        if scores.ndim != 3:
            raise ValueError(f"Expected scores to be 3D (B,N,M), got {tuple(scores.shape)}")
        #  Get batch size from scores
        batch_size = scores.shape[0]
        # Get number of points for each graph in the pair
        n_points = data_dict["ns"]

        # number of points for each graph in the pair, shape (B,)
        nrows = n_points[idx1].to(scores.device).view(-1)
        ncols = n_points[idx2].to(scores.device).view(-1)
        if nrows.numel() != batch_size or ncols.numel() != batch_size:
            raise ValueError(
                "n_points batch size mismatch: "
                f"scores={batch_size}, nrows={nrows.numel()}, ncols={ncols.numel()}"
            )
        # minimum number of points across the pair, shape (B,)
        min_point_tensor = torch.minimum(nrows, ncols).to(dtype=torch.float32)
        min_point_tensor_safe = torch.clamp(min_point_tensor, min=1.0)
        
        # Build GT correspondence counts for supervision (losses/logging).
        # Inference-time matching must not depend on these values.
        gt_perm_mat = data_dict.get("gt_perm_mat")
        if gt_perm_mat is None:
            gt_ks = torch.zeros(batch_size, dtype=torch.float32, device=scores.device)
        else:
            if gt_perm_mat.ndim != 3:
                raise ValueError(f"Expected gt_perm_mat to be 3D, got shape {tuple(gt_perm_mat.shape)}")
            if gt_perm_mat.shape[0] != batch_size:
                raise ValueError(
                    f"gt_perm_mat batch mismatch: expected {batch_size}, got {gt_perm_mat.shape[0]}"
                )

            max_n1 = int(nrows.max().item()) if batch_size > 0 else 0
            max_n2 = int(ncols.max().item()) if batch_size > 0 else 0
            if gt_perm_mat.shape[1] < max_n1 or gt_perm_mat.shape[2] < max_n2:
                raise ValueError(
                    "gt_perm_mat is smaller than requested n_points real block: "
                    f"gt_perm_mat={tuple(gt_perm_mat.shape)}, max_n1={max_n1}, max_n2={max_n2}"
                )

            gt_ks = torch.tensor(
                [
                    torch.sum(
                        gt_perm_mat[b, : int(nrows[b].item()), : int(ncols[b].item())]
                    )
                    for b in range(batch_size)
                ],
                dtype=torch.float32,
                device=scores.device,
            )

        # Only build no_pair_mask during training. During eval/inference we keep
        # matching decisions fully prediction-driven.
        no_pair_mask = None
        if self.training:
            if "label" in data_dict:
                labels = data_dict["label"].to(scores.device).view(-1).long()
                imposter_mask = labels == 0
                if imposter_mask.any():
                    gt_ks = gt_ks.clone()
                    gt_ks[imposter_mask] = 0.0
                no_pair_mask = imposter_mask | (gt_ks == 0)
            else:
                no_pair_mask = gt_ks == 0

        return min_point_tensor, min_point_tensor_safe, gt_ks, no_pair_mask

    def _predict_k(self, scores_for_k, n_points, idx1, idx2, min_point_tensor, gt_ks):
        min_point_tensor_safe = torch.clamp(min_point_tensor, min=1.0)

        if self.regression:
            print("Predicting K using AFAU")
            batch_size = scores_for_k.shape[0]
            dummy_row = self.univ_size - scores_for_k.shape[1]
            dummy_col = self.univ_size - scores_for_k.shape[2]
            if dummy_row < 0 or dummy_col < 0:
                raise ValueError(
                    f"AFAU univ_size={self.univ_size} is smaller than sinkhorn score size "
                    f"({scores_for_k.shape[1]}, {scores_for_k.shape[2]})"
                )

            nrows = n_points[idx1].to(scores_for_k.device).view(-1)
            ncols = n_points[idx2].to(scores_for_k.device).view(-1)
            max_n1 = int(torch.max(nrows).item()) if nrows.numel() > 0 else 0
            max_n2 = int(torch.max(ncols).item()) if ncols.numel() > 0 else 0

            # AFAU
            init_row_emb = torch.zeros(
                (batch_size, max_n1, self.univ_size),
                dtype=torch.float32,
                device=scores_for_k.device,
            )
            init_col_emb = torch.zeros(
                (batch_size, max_n2, self.univ_size),
                dtype=torch.float32,
                device=scores_for_k.device,
            )

            for b in range(batch_size):
                n2_b = int(ncols[b].item())
                if n2_b <= 0:
                    continue
                index = torch.arange(n2_b, dtype=torch.long, device=scores_for_k.device).unsqueeze(1)
                init_col_emb_one = torch.zeros(
                    max_n2,
                    self.univ_size,
                    dtype=torch.float32,
                    device=scores_for_k.device,
                ).scatter_(1, index, 1)
                init_col_emb[b] = init_col_emb_one

            scores_for_k_input = scores_for_k.detach() if self._should_detach_for_k() else scores_for_k
            out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, scores_for_k_input)

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
        else:
            k_logits = None
            ks = torch.where(
                min_point_tensor > 0,
                gt_ks / min_point_tensor_safe,
                torch.zeros_like(gt_ks),
            )

        return ks, k_logits

    @staticmethod
    def _align_auth_scalar_features(auth_scalar_features, target_dim):
        if auth_scalar_features.shape[1] == target_dim:
            return auth_scalar_features
        if auth_scalar_features.shape[1] > target_dim:
            return auth_scalar_features[:, :target_dim]
        pad = auth_scalar_features.new_zeros(
            auth_scalar_features.shape[0], target_dim - auth_scalar_features.shape[1]
        )
        return torch.cat([auth_scalar_features, pad], dim=1)

    @staticmethod
    def _transport_entropy(block):
        if block.numel() <= 1:
            return block.new_tensor(0.0)
        p = block.reshape(-1).clamp(min=1e-12)
        ent = -(p * torch.log(p)).sum()
        norm = torch.log(block.new_tensor(float(p.numel())))
        return ent / norm.clamp(min=1e-12)

    @staticmethod
    def _topk_confidence_gap(block):
        if block.numel() == 0:
            return block.new_tensor(0.0)
        n2 = block.shape[1]
        if n2 == 1:
            return block[:, 0].mean()
        top2 = torch.topk(block, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).mean()

    @staticmethod
    def _compute_pred_match_ratio(k_pred_count, min_point_tensor_safe):
        return torch.where(
            min_point_tensor_safe > 0,
            k_pred_count / min_point_tensor_safe,
            torch.zeros_like(k_pred_count),
        ).clamp(0.0, 1.0)

    def _compute_auth_scalar_features(self, pred_match_ratio, ks, full_transport, real_transport, nrows, ncols):
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
                transport_entropy.append(self._transport_entropy(p_real))
                topk_conf_gap.append(self._topk_confidence_gap(p_real))
            else:
                zero = p_full.new_tensor(0.0)
                transport_entropy.append(zero)
                topk_conf_gap.append(zero)

        dustbin_mass_row = torch.stack(dustbin_mass_row, dim=0).to(pred_match_ratio.dtype)
        dustbin_mass_col = torch.stack(dustbin_mass_col, dim=0).to(pred_match_ratio.dtype)
        transport_entropy = torch.stack(transport_entropy, dim=0).to(pred_match_ratio.dtype)
        topk_conf_gap = torch.stack(topk_conf_gap, dim=0).to(pred_match_ratio.dtype)

        auth_scalar_features = torch.stack(
            [
                pred_match_ratio,
                ks.view(-1),
                dustbin_mass_row,
                dustbin_mass_col,
                transport_entropy,
                topk_conf_gap,
            ],
            dim=1,
        )
        auth_scalar_features = self._align_auth_scalar_features(auth_scalar_features, self.auth_scalar_dim)
        return (
            auth_scalar_features,
            dustbin_mass_row,
            dustbin_mass_col,
            transport_entropy,
            topk_conf_gap,
        )

    def _pool_matcher_embedding(self, matcher_emb, nrows, ncols, max_n1, max_n2):
        if matcher_emb is None or matcher_emb.ndim != 3:
            return None

        pooled = []
        expected = max_n1 * max_n2
        for b in range(matcher_emb.shape[0]):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 <= 0 or n2 <= 0:
                pooled.append(matcher_emb.new_zeros(matcher_emb.shape[-1]))
                continue

            if matcher_emb.shape[1] != expected:
                # Fallback for unexpected flatten layout.
                if not self._auth_pool_warned:
                    logger.warning(
                        "Matcher embedding length mismatch (got %s, expected %s). Falling back to prefix pooling.",
                        matcher_emb.shape[1],
                        expected,
                    )
                    self._auth_pool_warned = True
                active = matcher_emb[b, : (n1 * n2)]
            else:
                mask = matcher_emb.new_zeros((max_n2, max_n1), dtype=torch.bool)
                mask[:n2, :n1] = True
                active = matcher_emb[b, mask.reshape(-1)]

            if active.numel() == 0:
                pooled.append(matcher_emb.new_zeros(matcher_emb.shape[-1]))
            else:
                pooled.append(active.mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _compute_auth_logits(self, auth_scalar_features, auth_pooled_features=None):
        if self.auth_use_pooled:
            pooled = auth_pooled_features
            if pooled is None:
                pooled = auth_scalar_features.new_zeros((auth_scalar_features.shape[0], self.auth_pooled_dim))
            auth_input = torch.cat([auth_scalar_features, pooled], dim=1)
        else:
            auth_input = auth_scalar_features
        auth_logit = self.auth_head(auth_input).squeeze(-1)
        return auth_logit, auth_input

    def _compute_transport(self, scores, nrows, ncols):
        if scores.ndim != 3:
            raise ValueError(f"Expected scores to be 3D (B,N,M), got {tuple(scores.shape)}")

        batch_size, max_n1, max_n2 = scores.shape
        nrows = nrows.to(scores.device).view(-1)
        ncols = ncols.to(scores.device).view(-1)
        if nrows.numel() != batch_size or ncols.numel() != batch_size:
            raise ValueError(
                "nrows/ncols batch mismatch for transport: "
                f"scores={batch_size}, nrows={nrows.numel()}, ncols={ncols.numel()}"
            )

        full_transport = scores.new_zeros(batch_size, max_n1 + 1, max_n2 + 1)
        real_transport = scores.new_zeros(batch_size, max_n1, max_n2)

        for b in range(batch_size):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 < 0 or n2 < 0:
                raise ValueError(f"Negative graph size detected: n1={n1}, n2={n2}")
            if n1 > max_n1 or n2 > max_n2:
                raise ValueError(
                    f"Requested active size exceeds padded score shape: "
                    f"n1={n1}, n2={n2}, max_n1={max_n1}, max_n2={max_n2}"
                )
            if n1 == 0 and n2 == 0:
                continue

            logits = scores[b, :n1, :n2]
            log_transport = log_optimal_transport(logits / self.tau, self.bin_score, SK_ITER_NUM)
            transport = torch.exp(log_transport)
            transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)

            full_transport[b, : n1 + 1, : n2 + 1] = transport
            if n1 > 0 and n2 > 0:
                real_transport[b, :n1, :n2] = transport[:n1, :n2]

        return full_transport, real_transport

    def _apply_topk(self, real_transport, k_match, nrows, ncols):
        _, topk_mask = soft_topk(
            real_transport,
            k_match.view(-1),
            SK_ITER_NUM,
            self.tau,
            nrows,
            ncols,
            True,
        )
        refined_real = topk_mask * real_transport
        refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)
        return refined_real

    def _build_permutation(self, refined_real, full_transport, k_match, nrows, ncols):
        x = hungarian(refined_real, nrows, ncols)
        top_indices = torch.argsort(
            x.mul(refined_real).reshape(x.shape[0], -1),
            descending=True,
            dim=-1,
        )
        x = torch.zeros_like(refined_real)
        x = greedy_perm(x, top_indices, k_match)

        perm_full = refined_real.new_zeros(
            refined_real.shape[0],
            refined_real.shape[1] + 1,
            refined_real.shape[2] + 1,
        )
        perm_full[:, : refined_real.shape[1], : refined_real.shape[2]] = x

        for b in range(perm_full.shape[0]):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 > 0:
                row_sums = x[b, :n1, :n2].sum(dim=1)
                perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
            if n2 > 0:
                col_sums = x[b, :n1, :n2].sum(dim=0)
                perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
            perm_full[b, n1, n2] = 0.0

        reject_rows = []
        reject_cols = []
        if getattr(self, "dustbin_reject_enable", False):
            margin = self._current_dustbin_reject_margin().to(
                full_transport.device,
                dtype=full_transport.dtype,
            )
            for b in range(perm_full.shape[0]):
                n1 = int(nrows[b].item())
                n2 = int(ncols[b].item())
                p = full_transport[b]
                row_reject = p.new_zeros(n1, dtype=torch.bool)
                col_reject = p.new_zeros(n2, dtype=torch.bool)

                if n1 > 0 and n2 > 0:
                    best_j = refined_real[b, :n1, :n2].argmax(dim=1)
                    best_p = p[torch.arange(n1, device=p.device), best_j]
                    dust_p = p[:n1, n2]
                    row_reject = dust_p >= (best_p + margin)

                    best_i = refined_real[b, :n1, :n2].argmax(dim=0)
                    best_p_col = p[best_i, torch.arange(n2, device=p.device)]
                    dust_p_col = p[n1, :n2]
                    col_reject = dust_p_col >= (best_p_col + margin)

                if n1 > 0:
                    x[b, :n1, :n2][row_reject] = 0.0
                if n2 > 0:
                    x[b, :n1, :n2][:, col_reject] = 0.0

                if n1 > 0:
                    row_sums = x[b, :n1, :n2].sum(dim=1)
                    perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
                if n2 > 0:
                    col_sums = x[b, :n1, :n2].sum(dim=0)
                    perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
                perm_full[b, :n1, :n2] = x[b, :n1, :n2]

                reject_rows.append(row_reject)
                reject_cols.append(col_reject)
        else:
            for b in range(perm_full.shape[0]):
                n1 = int(nrows[b].item())
                n2 = int(ncols[b].item())
                reject_rows.append(full_transport.new_zeros(n1, dtype=torch.bool))
                reject_cols.append(full_transport.new_zeros(n2, dtype=torch.bool))

        return x, perm_full, reject_rows, reject_cols

    def _dustbin_supervision_loss(self, pred, gt, nrows, ncols):
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

    def _compute_dustbin_losses(self, ds_mat, full_transport, gt_perm_mat, nrows, ncols):
        if gt_perm_mat is None:
            return 0.0, 0.0

        dustbin_loss = self._dustbin_supervision_loss(ds_mat, gt_perm_mat, nrows, ncols)
        dustbin_loss = dustbin_loss * self.dustbin_loss_weight

        eps = 1e-8
        sg_loss = ds_mat.new_tensor(0.0)
        sg_count = ds_mat.new_tensor(0.0)
        for b in range(full_transport.shape[0]):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 == 0 and n2 == 0:
                continue

            gt_block = gt_perm_mat[b, :n1, :n2]
            p = full_transport[b]

            if n1 > 0 and n2 > 0:
                match_mask = gt_block > 0.5
                if match_mask.any():
                    sg_loss -= torch.log(p[:n1, :n2].clamp(min=eps))[match_mask].sum()
                    sg_count += match_mask.sum()

            if n1 > 0:
                if n2 > 0:
                    row_unmatched = gt_block.sum(dim=1) == 0
                else:
                    row_unmatched = torch.ones(n1, dtype=torch.bool, device=p.device)
                if row_unmatched.any():
                    sg_loss -= torch.log(p[:n1, n2].clamp(min=eps))[row_unmatched].sum()
                    sg_count += row_unmatched.sum()

            if n2 > 0:
                if n1 > 0:
                    col_unmatched = gt_block.sum(dim=0) == 0
                else:
                    col_unmatched = torch.ones(n2, dtype=torch.bool, device=p.device)
                if col_unmatched.any():
                    sg_loss -= torch.log(p[n1, :n2].clamp(min=eps))[col_unmatched].sum()
                    sg_count += col_unmatched.sum()

        sg_dustbin_loss = sg_loss / sg_count if sg_count > 0 else ds_mat.new_tensor(0.0)
        return dustbin_loss, sg_dustbin_loss

    def _compute_dustbin_margin_loss(self, full_transport, gt_perm_mat, nrows, ncols):
        if gt_perm_mat is None:
            return full_transport.new_tensor(0.0)

        margin = self._current_dustbin_reject_margin().to(
            full_transport.device,
            dtype=full_transport.dtype,
        )
        temp = float(DUSTBIN_MARGIN_TEMP)
        loss = full_transport.new_tensor(0.0)
        count = full_transport.new_tensor(0.0)

        for b in range(full_transport.shape[0]):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 <= 0 and n2 <= 0:
                continue

            p = full_transport[b]
            gt_block = gt_perm_mat[b, :n1, :n2] if (n1 > 0 and n2 > 0) else None

            if n1 > 0 and n2 > 0:
                best_p_row = p[:n1, :n2].max(dim=1).values
            else:
                best_p_row = p.new_zeros(n1)
            dust_p_row = p[:n1, n2] if n1 > 0 else p.new_zeros(0)
            row_gate = torch.sigmoid((dust_p_row - best_p_row - margin) / temp)
            if n1 > 0:
                if gt_block is not None:
                    row_target = (gt_block.sum(dim=1) == 0).to(row_gate.dtype)
                else:
                    row_target = torch.ones_like(row_gate)
                loss += F.binary_cross_entropy(row_gate, row_target, reduction="sum")
                count += row_gate.numel()

            if n1 > 0 and n2 > 0:
                best_p_col = p[:n1, :n2].max(dim=0).values
            else:
                best_p_col = p.new_zeros(n2)
            dust_p_col = p[n1, :n2] if n2 > 0 else p.new_zeros(0)
            col_gate = torch.sigmoid((dust_p_col - best_p_col - margin) / temp)
            if n2 > 0:
                if gt_block is not None:
                    col_target = (gt_block.sum(dim=0) == 0).to(col_gate.dtype)
                else:
                    col_target = torch.ones_like(col_gate)
                loss += F.binary_cross_entropy(col_gate, col_target, reduction="sum")
                count += col_gate.numel()

        if count > 0:
            loss = loss / count
        return loss

    def _compute_k_match(self, ks, gt_ks, min_points, no_pair_mask, training, auth_prob=None):
        k_pred_count = torch.nan_to_num(
            ks.view(-1) * min_points,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        use_pred_k_train = bool(getattr(self, "train_use_pred_k", False))
        if training:
            if use_pred_k_train:
                alpha = float(getattr(self, "k_pred_ramp", 1.0))
                alpha = max(0.0, min(1.0, alpha))
                k_match = ((1.0 - alpha) * gt_ks) + (alpha * k_pred_count)
            else:
                k_match = gt_ks
        else:
            k_match = k_pred_count

        # ``no_pair_mask`` is a training-only supervision control.
        if training and no_pair_mask is not None:
            k_match = torch.where(no_pair_mask, torch.zeros_like(k_match), k_match)

        if not training:
            if self.k_gate_enable:
                gate_thresh = self._current_k_gate_thresh().to(ks.device, dtype=ks.dtype)
                k_match = torch.where(
                    ks.view(-1) < gate_thresh,
                    torch.zeros_like(k_match),
                    k_match,
                )
            if self.auth_gate_enable and auth_prob is not None:
                auth_gate_thresh = self._current_auth_gate_thresh().to(
                    auth_prob.device,
                    dtype=auth_prob.dtype,
                )
                k_match = torch.where(
                    auth_prob.view(-1) < auth_gate_thresh,
                    torch.zeros_like(k_match),
                    k_match,
                )
            if self.k_match_rounding == "floor":
                k_match = torch.floor(k_match)
            else:
                k_match = torch.round(k_match)

        k_match = torch.clamp(k_match, min=0.0)
        k_match = torch.minimum(k_match, min_points)
        return k_match, k_pred_count

    def _compute_k_losses(
        self,
        ks,
        k_logits,
        supervised_ks,
        k_pred_count,
        gt_ks,
        min_point_tensor,
        min_point_tensor_safe,
        data_dict,
    ):
        zero = ks.new_tensor(0.0)
        k_reg_loss = zero
        k_cls_loss = zero
        k_count_loss = zero
        k_gate_loss = zero
        k_gate_prob = None

        if self.regression:
            k_reg_loss = F.mse_loss(ks, supervised_ks) * self.k_factor
            if "label" in data_dict:
                labels = data_dict["label"].to(ks.device).view(-1).float()
                k_cls_loss = F.binary_cross_entropy_with_logits(k_logits.view(-1), labels)

            count_err = F.smooth_l1_loss(k_pred_count, gt_ks, reduction="none")
            count_err = count_err / min_point_tensor_safe
            k_count_loss = count_err.mean()

            ks_loss = (self.k_reg_weight * k_reg_loss) + (self.k_cls_weight * k_cls_loss) + k_count_loss
            ks_error = F.l1_loss(ks * min_point_tensor, gt_ks)
            if self._k_gate_learning_enabled() and "label" in data_dict:
                labels = data_dict["label"].to(ks.device).view(-1).float()
                k_gate_loss, k_gate_prob = self._compute_k_gate_loss(ks, labels)
        else:
            ks_loss = 0.0
            ks_error = 0.0

        return ks_loss, ks_error, k_reg_loss, k_cls_loss, k_count_loss, k_gate_loss, k_gate_prob

    def forward(self, data_dict, regression=True):
        del regression  # kept for interface compatibility

        branch = str(getattr(self, "match_branch", "topk")).strip().lower()
        if branch not in {"shared_matcher", "topk", "dustbin"}:
            branch = "topk"
        use_topk_branch = branch == "topk"
        use_dustbin_branch = branch == "dustbin"

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
            if log_nans:
                _nan_stats("node_layers", node_maps)

            edge_maps = self.edge_layers(node_maps)
            if log_nans:
                _nan_stats("edge_layers", edge_maps)

            global_feature = self.final_layers(edge_maps).reshape((node_maps.shape[0], -1))
            global_features.append(global_feature)
            if log_nans:
                _nan_stats("final_layers", global_feature)

            node_maps = normalize_over_channels(node_maps)
            edge_maps = normalize_over_channels(edge_maps)
            if log_nans:
                _nan_stats("norm_nodes", node_maps)
                _nan_stats("norm_edges", edge_maps)

            node_desc = concat_features(feature_align(node_maps, point, num_p, self.rescale), num_p)
            edge_desc = concat_features(feature_align(edge_maps, point, num_p, self.rescale), num_p)
            if log_nans:
                _nan_stats("feature_align_U", node_desc)
                _nan_stats("feature_align_F", edge_desc)

            node_features = torch.cat((node_desc, edge_desc), dim=1)
            if log_nans:
                _nan_stats("node_features_cat", node_features)

            pos = normalize_keypoints(point, image.shape)
            pos_emb = self.pos_mlp(pos).permute(0, 2, 1)
            pos_emb = concat_features(pos_emb, num_p)
            node_features = node_features + pos_emb
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            if log_nans:
                _nan_stats("message_pass_node_features", graph.x)

            edge_graph = self.build_edge_features_from_node_features(graph)
            if log_nans:
                edge_attr = edge_graph[0].edge_attr if isinstance(edge_graph, list) else edge_graph.edge_attr
                _nan_stats("build_edge_features_from_node_features", edge_attr)

            processed_graphs.append(edge_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1)
            for global_src, global_tgt in lexico_iter(global_features)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]
        if log_nans:
            _nan_stats("global_weights_list[0]", global_weights_list[0] if global_weights_list else None)

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]
        if log_nans and unary_affs_list:
            _nan_stats("unary_affs_list[0]", unary_affs_list[0][0] if unary_affs_list[0] else None)

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(processed_graphs), global_weights_list)
        ]
        if log_nans and quadratic_affs_list:
            _nan_stats("quadratic_affs_list[0]", quadratic_affs_list[0][0] if quadratic_affs_list[0] else None)
        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        if log_nans:
            self._nan_debug_done = True

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
        batch_size = match_scores.shape[0]
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
            print("Predicting K using AFAU")
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
            scores_for_k_input = ss_base.detach() if self._should_detach_for_k() else ss_base
            out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, scores_for_k_input)
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
            ks = torch.relu(k_logits)/(1.0 + torch.relu(k_logits))
        else:
            k_logits = None
            ks = torch.where(
                min_point_tensor > 0,
                gt_ks / min_point_tensor_safe,
                torch.zeros_like(gt_ks),
            )

        no_pair_mask_for_matching = no_pair_mask if self.training else None

        k_pred_count = torch.nan_to_num(
            ks.view(-1) * min_point_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        use_pred_k_train = bool(getattr(self, "train_use_pred_k", False))
        if self.training:
            if use_pred_k_train:
                alpha = float(getattr(self, "k_pred_ramp", 1.0))
                alpha = max(0.0, min(1.0, alpha))
                k_match = ((1.0 - alpha) * gt_ks) + (alpha * k_pred_count)
            else:
                k_match = gt_ks
        else:
            k_match = k_pred_count
        if self.training and no_pair_mask_for_matching is not None:
            k_match = torch.where(no_pair_mask_for_matching, torch.zeros_like(k_match), k_match)
        if not self.training:
            if self.k_gate_enable:
                gate_thresh = self._current_k_gate_thresh().to(ks.device, dtype=ks.dtype)
                k_match = torch.where(
                    ks.view(-1) < gate_thresh,
                    torch.zeros_like(k_match),
                    k_match,
                )
            if self.auth_gate_enable:
                pass
            if self.k_match_rounding == "floor":
                k_match = torch.floor(k_match)
            else:
                k_match = torch.round(k_match)
        k_match = torch.clamp(k_match, min=0.0)
        k_match = torch.minimum(k_match, min_point_tensor)

        if match_scores.ndim != 3:
            raise ValueError(f"Expected scores to be 3D (B,N,M), got {tuple(match_scores.shape)}")
        batch_size_transport, max_n1_transport, max_n2_transport = match_scores.shape
        nrows_transport = n_points[idx1].to(match_scores.device).view(-1)
        ncols_transport = n_points[idx2].to(match_scores.device).view(-1)
        if nrows_transport.numel() != batch_size_transport or ncols_transport.numel() != batch_size_transport:
            raise ValueError(
                "nrows/ncols batch mismatch for transport: "
                f"scores={batch_size_transport}, nrows={nrows_transport.numel()}, "
                f"ncols={ncols_transport.numel()}"
            )
        full_transport = match_scores.new_zeros(batch_size_transport, max_n1_transport + 1, max_n2_transport + 1)
        real_transport = match_scores.new_zeros(batch_size_transport, max_n1_transport, max_n2_transport)
        for b in range(batch_size_transport):
            n1 = int(nrows_transport[b].item())
            n2 = int(ncols_transport[b].item())
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

        if use_topk_branch:
            _, topk_mask = soft_topk(
                real_transport,
                k_match.view(-1),
                SK_ITER_NUM,
                self.tau,
                n_points[idx1],
                n_points[idx2],
                True,
            )
            refined_real = topk_mask * real_transport
            refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            refined_real = real_transport.clone()
            refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)

        ds_mat = full_transport.clone()
        ds_mat[:, : real_transport.shape[1], : real_transport.shape[2]] = refined_real

        x = hungarian(refined_real, n_points[idx1], n_points[idx2])
        top_indices = torch.argsort(
            x.mul(refined_real).reshape(x.shape[0], -1),
            descending=True,
            dim=-1,
        )
        x = torch.zeros_like(refined_real)
        match_count_for_perm = k_match if use_topk_branch else min_point_tensor
        x = greedy_perm(x, top_indices, match_count_for_perm)
        perm_full = refined_real.new_zeros(
            refined_real.shape[0],
            refined_real.shape[1] + 1,
            refined_real.shape[2] + 1,
        )
        perm_full[:, : refined_real.shape[1], : refined_real.shape[2]] = x
        for b in range(perm_full.shape[0]):
            n1 = int(n_points[idx1][b].item())
            n2 = int(n_points[idx2][b].item())
            if n1 > 0:
                row_sums = x[b, :n1, :n2].sum(dim=1)
                perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
            if n2 > 0:
                col_sums = x[b, :n1, :n2].sum(dim=0)
                perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
            perm_full[b, n1, n2] = 0.0
        reject_rows = []
        reject_cols = []
        if getattr(self, "dustbin_reject_enable", False):
            margin = self._current_dustbin_reject_margin().to(
                full_transport.device,
                dtype=full_transport.dtype,
            )
            for b in range(perm_full.shape[0]):
                n1 = int(n_points[idx1][b].item())
                n2 = int(n_points[idx2][b].item())
                p = full_transport[b]
                row_reject = p.new_zeros(n1, dtype=torch.bool)
                col_reject = p.new_zeros(n2, dtype=torch.bool)
                if n1 > 0 and n2 > 0:
                    best_j = refined_real[b, :n1, :n2].argmax(dim=1)
                    best_p = p[torch.arange(n1, device=p.device), best_j]
                    dust_p = p[:n1, n2]
                    row_reject = dust_p >= (best_p + margin)
                    best_i = refined_real[b, :n1, :n2].argmax(dim=0)
                    best_p_col = p[best_i, torch.arange(n2, device=p.device)]
                    dust_p_col = p[n1, :n2]
                    col_reject = dust_p_col >= (best_p_col + margin)
                if n1 > 0:
                    x[b, :n1, :n2][row_reject] = 0.0
                if n2 > 0:
                    x[b, :n1, :n2][:, col_reject] = 0.0
                if n1 > 0:
                    row_sums = x[b, :n1, :n2].sum(dim=1)
                    perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
                if n2 > 0:
                    col_sums = x[b, :n1, :n2].sum(dim=0)
                    perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
                perm_full[b, :n1, :n2] = x[b, :n1, :n2]
                reject_rows.append(row_reject)
                reject_cols.append(col_reject)
        else:
            for b in range(perm_full.shape[0]):
                n1 = int(n_points[idx1][b].item())
                n2 = int(n_points[idx2][b].item())
                reject_rows.append(full_transport.new_zeros(n1, dtype=torch.bool))
                reject_cols.append(full_transport.new_zeros(n2, dtype=torch.bool))

        supervised_ks = torch.where(
            min_point_tensor > 0,
            gt_ks / min_point_tensor_safe,
            torch.zeros_like(gt_ks),
        )
        if no_pair_mask is not None:
            supervised_ks = torch.where(no_pair_mask, torch.zeros_like(supervised_ks), supervised_ks)

        # Authentication features/logits must stay prediction-driven only.
        pred_match_ratio = self._compute_pred_match_ratio(k_pred_count, min_point_tensor_safe)
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
                transport_entropy.append(self._transport_entropy(p_real))
                topk_conf_gap.append(self._topk_confidence_gap(p_real))
            else:
                zero = p_full.new_tensor(0.0)
                transport_entropy.append(zero)
                topk_conf_gap.append(zero)
        dustbin_mass_row = torch.stack(dustbin_mass_row, dim=0).to(pred_match_ratio.dtype)
        dustbin_mass_col = torch.stack(dustbin_mass_col, dim=0).to(pred_match_ratio.dtype)
        transport_entropy = torch.stack(transport_entropy, dim=0).to(pred_match_ratio.dtype)
        topk_conf_gap = torch.stack(topk_conf_gap, dim=0).to(pred_match_ratio.dtype)
        auth_features_scalar = torch.stack(
            [
                pred_match_ratio,
                ks.view(-1),
                dustbin_mass_row,
                dustbin_mass_col,
                transport_entropy,
                topk_conf_gap,
            ],
            dim=1,
        )
        auth_features_scalar = self._align_auth_scalar_features(auth_features_scalar, self.auth_scalar_dim)

        auth_features_pooled = None
        if self.auth_use_pooled:
            pooled_raw = None
            if matcher_emb is not None and matcher_emb.ndim == 3:
                pooled = []
                expected = max_n1 * max_n2
                for b in range(matcher_emb.shape[0]):
                    n1 = int(nrows[b].item())
                    n2 = int(ncols[b].item())
                    if n1 <= 0 or n2 <= 0:
                        pooled.append(matcher_emb.new_zeros(matcher_emb.shape[-1]))
                        continue
                    if matcher_emb.shape[1] != expected:
                        if not self._auth_pool_warned:
                            logger.warning(
                                "Matcher embedding length mismatch (got %s, expected %s). "
                                "Falling back to prefix pooling.",
                                matcher_emb.shape[1],
                                expected,
                            )
                            self._auth_pool_warned = True
                        active = matcher_emb[b, : (n1 * n2)]
                    else:
                        mask = matcher_emb.new_zeros((max_n2, max_n1), dtype=torch.bool)
                        mask[:n2, :n1] = True
                        active = matcher_emb[b, mask.reshape(-1)]
                    if active.numel() == 0:
                        pooled.append(matcher_emb.new_zeros(matcher_emb.shape[-1]))
                    else:
                        pooled.append(active.mean(dim=0))
                pooled_raw = torch.stack(pooled, dim=0)
            if pooled_raw is not None:
                auth_features_pooled = self.auth_pool_proj(pooled_raw)
            else:
                auth_features_pooled = auth_features_scalar.new_zeros(
                    (auth_features_scalar.shape[0], self.auth_pooled_dim)
                )
        else:
            auth_features_pooled = auth_features_scalar.new_zeros((auth_features_scalar.shape[0], 0))

        if self.auth_use_pooled:
            pooled_features = auth_features_pooled
            if pooled_features is None:
                pooled_features = auth_features_scalar.new_zeros(
                    (auth_features_scalar.shape[0], self.auth_pooled_dim)
                )
            auth_input = torch.cat([auth_features_scalar, pooled_features], dim=1)
        else:
            auth_input = auth_features_scalar
        auth_logit = self.auth_head(auth_input).squeeze(-1)
        auth_prob_raw = torch.sigmoid(auth_logit)
        hybrid_prob = 0.5 * auth_prob_raw + 0.5 * pred_match_ratio.view(-1)
        auth_prob = hybrid_prob
        auth_gate_loss = auth_prob.new_tensor(0.0)
        auth_gate_prob = torch.zeros_like(auth_prob)
        if self.auth_gate_enable and self._k_gate_learning_enabled() and "label" in data_dict:
            labels = data_dict["label"].to(auth_prob.device).view(-1).float()
            auth_gate_loss, auth_gate_prob = self._compute_auth_gate_loss(auth_prob, labels)

        if use_topk_branch and (not self.training) and self.auth_gate_enable:
            auth_gate_thresh = self._current_auth_gate_thresh().to(
                auth_prob.device,
                dtype=auth_prob.dtype,
            )
            if (auth_prob.view(-1) < auth_gate_thresh).any():
                k_pred_count = torch.nan_to_num(
                    ks.view(-1) * min_point_tensor,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                k_match = k_pred_count
                if self.k_gate_enable:
                    gate_thresh = self._current_k_gate_thresh().to(ks.device, dtype=ks.dtype)
                    k_match = torch.where(
                        ks.view(-1) < gate_thresh,
                        torch.zeros_like(k_match),
                        k_match,
                    )
                auth_gate_thresh = self._current_auth_gate_thresh().to(
                    auth_prob.device,
                    dtype=auth_prob.dtype,
                )
                k_match = torch.where(
                    auth_prob.view(-1) < auth_gate_thresh,
                    torch.zeros_like(k_match),
                    k_match,
                )
                if self.k_match_rounding == "floor":
                    k_match = torch.floor(k_match)
                else:
                    k_match = torch.round(k_match)
                k_match = torch.clamp(k_match, min=0.0)
                k_match = torch.minimum(k_match, min_point_tensor)
                _, topk_mask = soft_topk(
                    real_transport,
                    k_match.view(-1),
                    SK_ITER_NUM,
                    self.tau,
                    n_points[idx1],
                    n_points[idx2],
                    True,
                )
                refined_real = topk_mask * real_transport
                refined_real = torch.nan_to_num(refined_real, nan=0.0, posinf=0.0, neginf=0.0)
                ds_mat = full_transport.clone()
                ds_mat[:, : real_transport.shape[1], : real_transport.shape[2]] = refined_real
                x = hungarian(refined_real, n_points[idx1], n_points[idx2])
                top_indices = torch.argsort(
                    x.mul(refined_real).reshape(x.shape[0], -1),
                    descending=True,
                    dim=-1,
                )
                x = torch.zeros_like(refined_real)
                x = greedy_perm(x, top_indices, k_match)
                perm_full = refined_real.new_zeros(
                    refined_real.shape[0],
                    refined_real.shape[1] + 1,
                    refined_real.shape[2] + 1,
                )
                perm_full[:, : refined_real.shape[1], : refined_real.shape[2]] = x
                for b in range(perm_full.shape[0]):
                    n1 = int(n_points[idx1][b].item())
                    n2 = int(n_points[idx2][b].item())
                    if n1 > 0:
                        row_sums = x[b, :n1, :n2].sum(dim=1)
                        perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
                    if n2 > 0:
                        col_sums = x[b, :n1, :n2].sum(dim=0)
                        perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
                    perm_full[b, n1, n2] = 0.0
                reject_rows = []
                reject_cols = []
                if getattr(self, "dustbin_reject_enable", False):
                    margin = self._current_dustbin_reject_margin().to(
                        full_transport.device,
                        dtype=full_transport.dtype,
                    )
                    for b in range(perm_full.shape[0]):
                        n1 = int(n_points[idx1][b].item())
                        n2 = int(n_points[idx2][b].item())
                        p = full_transport[b]
                        row_reject = p.new_zeros(n1, dtype=torch.bool)
                        col_reject = p.new_zeros(n2, dtype=torch.bool)
                        if n1 > 0 and n2 > 0:
                            best_j = refined_real[b, :n1, :n2].argmax(dim=1)
                            best_p = p[torch.arange(n1, device=p.device), best_j]
                            dust_p = p[:n1, n2]
                            row_reject = dust_p >= (best_p + margin)
                            best_i = refined_real[b, :n1, :n2].argmax(dim=0)
                            best_p_col = p[best_i, torch.arange(n2, device=p.device)]
                            dust_p_col = p[n1, :n2]
                            col_reject = dust_p_col >= (best_p_col + margin)
                        if n1 > 0:
                            x[b, :n1, :n2][row_reject] = 0.0
                        if n2 > 0:
                            x[b, :n1, :n2][:, col_reject] = 0.0
                        if n1 > 0:
                            row_sums = x[b, :n1, :n2].sum(dim=1)
                            perm_full[b, :n1, n2] = (row_sums == 0).to(perm_full.dtype)
                        if n2 > 0:
                            col_sums = x[b, :n1, :n2].sum(dim=0)
                            perm_full[b, n1, :n2] = (col_sums == 0).to(perm_full.dtype)
                        perm_full[b, :n1, :n2] = x[b, :n1, :n2]
                        reject_rows.append(row_reject)
                        reject_cols.append(col_reject)
                else:
                    for b in range(perm_full.shape[0]):
                        n1 = int(n_points[idx1][b].item())
                        n2 = int(n_points[idx2][b].item())
                        reject_rows.append(full_transport.new_zeros(n1, dtype=torch.bool))
                        reject_cols.append(full_transport.new_zeros(n2, dtype=torch.bool))

        zero = ks.new_tensor(0.0)
        k_reg_loss = zero
        k_cls_loss = zero
        k_count_loss = zero
        k_gate_loss = zero
        k_gate_prob = None
        if use_topk_branch and self.regression:
            k_reg_loss = F.mse_loss(ks, supervised_ks) * self.k_factor
            if "label" in data_dict:
                labels = data_dict["label"].to(ks.device).view(-1).float()
                k_cls_loss = F.binary_cross_entropy_with_logits(k_logits.view(-1), labels)
            count_err = F.smooth_l1_loss(k_pred_count, gt_ks, reduction="none")
            count_err = count_err / min_point_tensor_safe
            k_count_loss = count_err.mean()
            ks_loss = (self.k_reg_weight * k_reg_loss) + (self.k_cls_weight * k_cls_loss) + k_count_loss
            ks_error = F.l1_loss(ks * min_point_tensor, gt_ks)
            if self._k_gate_learning_enabled() and "label" in data_dict:
                labels = data_dict["label"].to(ks.device).view(-1).float()
                k_gate_loss, k_gate_prob = self._compute_k_gate_loss(ks, labels)
        else:
            ks_loss = zero
            ks_error = zero
            k_reg_loss = zero
            k_cls_loss = zero
            k_count_loss = zero
            k_gate_loss = zero
            k_gate_prob = torch.zeros_like(ks)

        if use_dustbin_branch and data_dict.get("gt_perm_mat") is not None:
            dustbin_loss = self._dustbin_supervision_loss(
                ds_mat,
                data_dict["gt_perm_mat"],
                n_points[idx1],
                n_points[idx2],
            )
            dustbin_loss = dustbin_loss * self.dustbin_loss_weight
            eps = 1e-8
            sg_loss = ds_mat.new_tensor(0.0)
            sg_count = ds_mat.new_tensor(0.0)
            for b in range(full_transport.shape[0]):
                n1 = int(n_points[idx1][b].item())
                n2 = int(n_points[idx2][b].item())
                if n1 == 0 and n2 == 0:
                    continue
                gt_block = data_dict["gt_perm_mat"][b, :n1, :n2]
                p = full_transport[b]
                if n1 > 0 and n2 > 0:
                    match_mask = gt_block > 0.5
                    if match_mask.any():
                        sg_loss -= torch.log(p[:n1, :n2].clamp(min=eps))[match_mask].sum()
                        sg_count += match_mask.sum()
                if n1 > 0:
                    if n2 > 0:
                        row_unmatched = gt_block.sum(dim=1) == 0
                    else:
                        row_unmatched = torch.ones(n1, dtype=torch.bool, device=p.device)
                    if row_unmatched.any():
                        sg_loss -= torch.log(p[:n1, n2].clamp(min=eps))[row_unmatched].sum()
                        sg_count += row_unmatched.sum()
                if n2 > 0:
                    if n1 > 0:
                        col_unmatched = gt_block.sum(dim=0) == 0
                    else:
                        col_unmatched = torch.ones(n2, dtype=torch.bool, device=p.device)
                    if col_unmatched.any():
                        sg_loss -= torch.log(p[n1, :n2].clamp(min=eps))[col_unmatched].sum()
                        sg_count += col_unmatched.sum()
            sg_dustbin_loss = sg_loss / sg_count if sg_count > 0 else ds_mat.new_tensor(0.0)
        else:
            dustbin_loss, sg_dustbin_loss = zero, zero

        dustbin_margin_loss = ds_mat.new_tensor(0.0)
        if use_dustbin_branch and self.dustbin_reject_enable and self._k_gate_learning_enabled():
            if data_dict.get("gt_perm_mat") is not None:
                margin = self._current_dustbin_reject_margin().to(
                    full_transport.device,
                    dtype=full_transport.dtype,
                )
                temp = float(DUSTBIN_MARGIN_TEMP)
                loss = full_transport.new_tensor(0.0)
                count = full_transport.new_tensor(0.0)
                for b in range(full_transport.shape[0]):
                    n1 = int(n_points[idx1][b].item())
                    n2 = int(n_points[idx2][b].item())
                    if n1 <= 0 and n2 <= 0:
                        continue
                    p = full_transport[b]
                    gt_block = data_dict["gt_perm_mat"][b, :n1, :n2] if (n1 > 0 and n2 > 0) else None
                    if n1 > 0 and n2 > 0:
                        best_p_row = p[:n1, :n2].max(dim=1).values
                    else:
                        best_p_row = p.new_zeros(n1)
                    dust_p_row = p[:n1, n2] if n1 > 0 else p.new_zeros(0)
                    row_gate = torch.sigmoid((dust_p_row - best_p_row - margin) / temp)
                    if n1 > 0:
                        if gt_block is not None:
                            row_target = (gt_block.sum(dim=1) == 0).to(row_gate.dtype)
                        else:
                            row_target = torch.ones_like(row_gate)
                        loss += F.binary_cross_entropy(row_gate, row_target, reduction="sum")
                        count += row_gate.numel()
                    if n1 > 0 and n2 > 0:
                        best_p_col = p[:n1, :n2].max(dim=0).values
                    else:
                        best_p_col = p.new_zeros(n2)
                    dust_p_col = p[n1, :n2] if n2 > 0 else p.new_zeros(0)
                    col_gate = torch.sigmoid((dust_p_col - best_p_col - margin) / temp)
                    if n2 > 0:
                        if gt_block is not None:
                            col_target = (gt_block.sum(dim=0) == 0).to(col_gate.dtype)
                        else:
                            col_target = torch.ones_like(col_gate)
                        loss += F.binary_cross_entropy(col_gate, col_target, reduction="sum")
                        count += col_gate.numel()
                if count > 0:
                    loss = loss / count
                dustbin_margin_loss = loss

        if use_dustbin_branch:
            ds_out = ds_mat
            perm_out = perm_full
            ns_out = [n_points[idx1] + 1, n_points[idx2] + 1]
            has_dustbin = True
            k_pred_ratio_out = pred_match_ratio if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_pred_count_out = k_pred_count if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_match_count_out = k_match if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_prob_out = ks if use_topk_branch else torch.zeros_like(min_point_tensor)
            if k_logits is None:
                k_logit_out = torch.zeros_like(min_point_tensor)
            elif use_topk_branch:
                k_logit_out = k_logits
            else:
                k_logit_out = torch.zeros_like(min_point_tensor)
        else:
            ds_out = refined_real
            perm_out = x
            ns_out = [n_points[idx1], n_points[idx2]]
            has_dustbin = False
            reject_rows = [refined_real.new_zeros(0, dtype=torch.bool) for _ in range(refined_real.shape[0])]
            reject_cols = [refined_real.new_zeros(0, dtype=torch.bool) for _ in range(refined_real.shape[0])]
            k_pred_ratio_out = pred_match_ratio if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_pred_count_out = k_pred_count if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_match_count_out = k_match if use_topk_branch else torch.zeros_like(min_point_tensor)
            k_prob_out = ks if use_topk_branch else torch.zeros_like(min_point_tensor)
            if k_logits is None:
                k_logit_out = torch.zeros_like(min_point_tensor)
            elif use_topk_branch:
                k_logit_out = k_logits
            else:
                k_logit_out = torch.zeros_like(min_point_tensor)

        data_dict.update(
            {
                "ds_mat": ds_out,
                "perm_mat": perm_out,
                "ks_loss": ks_loss,
                "ks_error": ks_error,
                "dustbin_loss": dustbin_loss,
                "sg_dustbin_loss": sg_dustbin_loss,
                "gt_ks": gt_ks,
                "k_pred_ratio": k_pred_ratio_out,
                "k_pred_count": k_pred_count_out,
                "k_match_count": k_match_count_out,
                "k_count_loss": k_count_loss,
                "k_prob": k_prob_out,
                "k_logit": k_logit_out,
                "k_reg_loss": k_reg_loss,
                "k_cls_loss": k_cls_loss,
                "k_gate_loss": k_gate_loss,
                "k_gate_prob": k_gate_prob if k_gate_prob is not None else torch.zeros_like(ks),
                "k_gate_thresh_value": self._current_k_gate_thresh().detach().to(ks.device),
                "k_gate_loss_weight": float(K_GATE_LOSS_WEIGHT),
                "auth_gate_loss": auth_gate_loss,
                "auth_gate_prob": auth_gate_prob,
                "auth_gate_thresh_value": self._current_auth_gate_thresh().detach().to(auth_prob.device),
                "auth_gate_loss_weight": float(AUTH_GATE_LOSS_WEIGHT),
                "dustbin_margin_loss": dustbin_margin_loss,
                "dustbin_margin_value": self._current_dustbin_reject_margin().detach().to(ds_mat.device),
                "dustbin_margin_loss_weight": float(DUSTBIN_MARGIN_LOSS_WEIGHT),
                "auth_logit": auth_logit,
                "auth_prob": auth_prob,
                "auth_prob_raw": auth_prob_raw,
                "hybrid_prob": hybrid_prob,
                "auth_features_scalar": auth_features_scalar,
                "auth_features_pooled": auth_features_pooled,
                "transport_entropy": transport_entropy,
                "dustbin_mass_row": dustbin_mass_row,
                "dustbin_mass_col": dustbin_mass_col,
                "topk_conf_gap": topk_conf_gap,
                "rejected_rows": reject_rows,
                "rejected_cols": reject_cols,
                "ns": ns_out,
                "has_dustbin": has_dustbin,
            }
        )

        return data_dict
