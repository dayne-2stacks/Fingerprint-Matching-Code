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
FEATURE_CHANNEL_NODE = 256 # ResNet34 layer3 channels
FEATURE_CHANNEL_EDGE = 512  # ResNet34 layer4 channels
# FEATURE_CHANNEL_NODE = 56
# FEATURE_CHANNEL_EDGE = 448
NODE_FEATURE_DIM = FEATURE_CHANNEL_NODE + FEATURE_CHANNEL_EDGE  # 1392
GLOBAL_FEATURE_DIM = FEATURE_CHANNEL_EDGE  # 1280
GLOBAL_STATE_DIM = GLOBAL_FEATURE_DIM * 2  # 2560

FIRST_ORDER = True
POSITIVE_EDGES = True
GNN_LAYER = 3
# SK_TAU= 0.005
SK_TAU = 0.01
SK_EMB = 1
GNN_FEAT = [16, 16, 16]
EDGE_EMB = False
BATCH_SIZE = 4

UNIV_SIZE = 600
SK_ITER_NUM = 25
SK_EPSILON = 1e-10
K_FACTOR = 5.0
RESCALE = (320, 240)
CROPSIZE = (240, 240)



def _compute_k_match_count(
    k_pred_count,
    gt_ks,
    min_points,
    *,
    training,
    no_pair_mask,
):
    if training:
        k_match = gt_ks
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

        self.k_embedding_dim = self.encoder_k.model_params["embedding_dim"]

        self.k_row_init = nn.Sequential(
            nn.Linear(self.message_pass_node_features.num_node_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.k_embedding_dim),
        )

        self.k_col_init = nn.Sequential(
            nn.Linear(self.message_pass_node_features.num_node_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.k_embedding_dim),
        )


    def forward(self, data_dict, stage=None):

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
  
        # List for global features and graphs
        global_features = []
        processed_graphs = []
        graph_node_descs = []

        # For each graph
        for image, point, num_p, graph in zip(images, points, n_points, graphs):
            # get the image
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Load node, edge and global feature maps 
            node_maps = self.node_layers(image)

            edge_maps = self.edge_layers(node_maps)

            global_feature = self.final_layers(edge_maps).reshape((node_maps.shape[0], -1))
            global_features.append(global_feature)

            # Normalize node and edge features over channels
            node_maps = normalize_over_channels(node_maps)
            edge_maps = normalize_over_channels(edge_maps)
            
            # interpolate with points
            node_desc = concat_features(feature_align(node_maps, point, num_p, self.cropsize), num_p)
            edge_desc = concat_features(feature_align(edge_maps, point, num_p, self.cropsize), num_p)

            node_features = torch.cat((node_desc, edge_desc), dim=1)
            
            # Add positional embedding
            # pos = normalize_keypoints(point, image.shape)
            # pos_emb = self.pos_mlp(pos).permute(0, 2, 1)
            # pos_emb = concat_features(pos_emb, num_p)
            # node_features = node_features + pos_emb
            graph.x = node_features
            # Create edge features from node features using SplineConv
            graph = self.message_pass_node_features(graph)
            edge_graph = self.build_edge_features_from_node_features(graph)
            processed_graphs.append(edge_graph)


            padded = torch.stack(pad_tensor([item.x for item in edge_graph]), dim=0)
            graph_node_descs.append(padded)


        # Get flobal features and compute affinities for each pair of graphs
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

                # Run GNN layers
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, "gnn_layer_{}".format(i))
                    tmp_emb = gnn_layer(adj, tmp_emb, n_points[idx1], n_points[idx2], b)
                qap_emb.append(tmp_emb.squeeze(0))

            matcher_emb = torch.stack(pad_tensor(qap_emb), dim=0)
            logits = self.classifier(matcher_emb)
            match_scores = logits.view(logits.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)


            # Doubly stochastic matrix for K head (AFAU path)
            ss_base = self.sinkhorn(match_scores, n_points[idx1], n_points[idx2], dummy_row=True)

            batch_size, max_n1, max_n2 = match_scores.shape
            nrows = n_points[idx1].to(match_scores.device).view(-1)
            ncols = n_points[idx2].to(match_scores.device).view(-1)

            min_point_tensor = torch.minimum(nrows, ncols).to(dtype=torch.float32)
            
            
            gt_perm_mat = data_dict.get("gt_perm_mat")
            if gt_perm_mat is None:
                gt_ks = torch.zeros(batch_size, dtype=torch.float32, device=match_scores.device)
            else:
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
                # init_row_emb = torch.zeros(
                #     (batch_size, max_n1_k, self.univ_size),
                #     dtype=torch.float32,
                #     device=ss_base.device,
                # )
                # init_col_emb = torch.zeros(
                #     (batch_size, max_n2_k, self.univ_size),
                #     dtype=torch.float32,
                #     device=ss_base.device,
                # )

                src_desc = graph_node_descs[idx1][:, :max_n1_k, :]
                tgt_desc = graph_node_descs[idx2][:, :max_n2_k, :]

                init_row_emb = self.k_row_init(src_desc)
                init_col_emb = self.k_col_init(tgt_desc)

                if self.training: 
                    data_dict["src_embeddings"] = src_desc.detach()
                    data_dict["tgt_embeddings"] = tgt_desc.detach()

                    
                for b in range(batch_size):
                    n2_b = int(ncols[b].item())
                    if n2_b <= 0:
                        continue
                    # index = torch.arange(n2_b, dtype=torch.long, device=ss_base.device).unsqueeze(1)
                    # init_col_emb_one = torch.zeros(
                    #     max_n2_k,
                    #     self.univ_size,
                    #     dtype=torch.float32,
                    #     device=ss_base.device,
                    # ).scatter_(1, index, 1)
                    # init_col_emb[b] = init_col_emb_one
                out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss_base.detach())
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



            # ----- Transport Matrices and no match probability -----
            # transport_with_dustbin keeps the extra row/col;
            transport_with_dustbin = match_scores.new_zeros(batch_size, max_n1 + 1, max_n2 + 1)
            no_match_prob = match_scores.new_zeros(batch_size)
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
                transport_with_dustbin[b, : n1 + 1, : n2 + 1] =torch.exp(
                        log_optimal_transport(
                            match_scores[b, :n1, :n2] / self.tau,
                            self.bin_score,
                            SK_ITER_NUM,
                        )
                    )
                # Mean reject probability for rows and columns.
                row_reject_mean = transport_with_dustbin[b, :n1, n2].mean()
                col_reject_mean = transport_with_dustbin[b, n1, :n2].mean()
                no_match_prob[b] = 0.5 * (row_reject_mean + col_reject_mean)

            no_match_prob = torch.clamp(no_match_prob, 0.0, 1.0)
            no_match_keep = 1.0 - no_match_prob

            # ----- K Match Count Selection (gated by no-match) -----
            predicted_match_count_raw = ks.view(-1) * min_point_tensor
            predicted_match_count = (no_match_keep.detach()) * predicted_match_count_raw

            if self.training:
                selected_match_count = torch.clamp(gt_ks, min=0.0)
            else:
                selected_match_count = torch.round(predicted_match_count)

            selected_match_count = torch.minimum(selected_match_count, min_point_tensor)

            if stage == 1:
                # Stage 1 forces GT-k as the selected match count.
                selected_match_count = torch.clamp(gt_ks, min=0.0)
                selected_match_count = torch.minimum(selected_match_count, min_point_tensor)


            # ----- Top-k Refinement Path -----
            
            topk_match_count = selected_match_count.view(-1)
            _, topk_refined_transport = soft_topk(
                ss_base,
                topk_match_count,
                SK_ITER_NUM,
                self.tau,
                n_points[idx1],
                n_points[idx2],
                return_prob=True,
            )
          


            if stage >= 3 or stage is None:
                # ----- Soft Dustbin Gate for Top-k -----
                # Same shape as topk_refined_transport: [B, max_n1, max_n2]
                dustbin_soft_gate = topk_refined_transport.new_ones(topk_refined_transport.shape)

                for b in range(topk_refined_transport.shape[0]):
                    n1 = int(nrows[b].item())
                    n2 = int(ncols[b].item())
                    if n1 <= 0 or n2 <= 0:
                        continue

                    # Reject probs from explicit dustbin transport
                    row_reject = transport_with_dustbin[b, :n1, n2].clamp(0.0, 1.0)  # P(row -> dustbin)
                    col_reject = transport_with_dustbin[b, n1, :n2].clamp(0.0, 1.0)  # P(col -> dustbin)

                    # Keep probs
                    row_keep = 1.0 - row_reject
                    col_keep = 1.0 - col_reject

                    # Outer product gate
                    gate = row_keep.unsqueeze(1) * col_keep.unsqueeze(0)
                    dustbin_soft_gate[b, :n1, :n2] = gate

                # Soft gated top-k (differentiable)
                topk_gated_soft = topk_refined_transport * dustbin_soft_gate

                # Publish the same score field used to build the final permutation so
                # downstream consumers of ds_mat see scores consistent with perm_mat.
                ds_mat = transport_with_dustbin.clone()
                ds_mat[:, : topk_gated_soft.shape[1], : topk_gated_soft.shape[2]] = topk_gated_soft
            else:
                ds_mat = topk_refined_transport
                topk_gated_soft = topk_refined_transport

            # ----- Top-k Permutation Proposal -----
            perm_topk_match_count = selected_match_count
            hungarian_seed_perm = hungarian(topk_gated_soft, n_points[idx1], n_points[idx2])
            top_indices = torch.argsort(
                hungarian_seed_perm.mul(topk_gated_soft).reshape(hungarian_seed_perm.shape[0], -1),
                descending=True,
                dim=-1,
            )
        
            perm_mat = torch.zeros_like(topk_refined_transport)
            perm_mat = greedy_perm(perm_mat, top_indices, perm_topk_match_count)


            # ----- Final Composition -----
            supervised_ks = torch.where(
                min_point_tensor > 0,
                gt_ks / min_point_tensor,
                torch.zeros_like(gt_ks),
            )
            if no_pair_mask is not None:
                supervised_ks = torch.where(no_pair_mask, torch.zeros_like(supervised_ks), supervised_ks)

        
            
            if bool(self.regression) and k_trainable:
                ks_loss = F.mse_loss(ks, supervised_ks) * self.k_factor
                # ks_loss = F.mse_loss(ks, gt_ks) * self.k_factor
                ks_error = F.l1_loss(ks * min_point_tensor, gt_ks)
                # ks_error = F.l1_loss(ks, gt_ks)
                ks_loss = ks_loss
            else:
                ks_loss = ks.new_tensor(0.0)
                ks_error = ks.new_tensor(0.0)



        # print(ks_error * 0.0001)

        data_dict.update(
            {   
                "ds_mat_dustbin": transport_with_dustbin,
                "ds_mat": ds_mat,
                "perm_mat": perm_mat,
                "ks_loss": ks_loss,
                "ks_error": ks_error,
                "gt_ks": gt_ks,
                "k_pred_count": predicted_match_count.detach(),
                "k_match_count": selected_match_count.detach(),
                "k_prob": ks,
                "ns": [n_points[idx1] + 1, n_points[idx2] + 1],
                "has_dustbin": True,
            }
        )

        return data_dict
