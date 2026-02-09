import logging
from src.model.feature_extractor import ResNet18_final as CNN
from src.model.spline_conv import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.feature_align import feature_align
from src.model.affinity_layer import InnerProductWithWeightsAffinity
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pad_tensor import pad_tensor
from utils.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from src.model.gnn import PYGNNLayer
from src.model.sinkhorn import Sinkhorn
from src.model.soft_topk import soft_topk, greedy_perm
from utils.hungarian import hungarian
from src.model.afau import Encoder

from utils.visualize import *

import itertools
from torch_sparse import spmm, SparseTensor
import yaml

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='fp.log', 
#     # Remove `encoding='utf-8'` if you have Python < 3.9 or if it causes issues
#     # encoding='utf-8',  
#     level=logging.DEBUG
# )


# Params
FEATURE_CHANNEL_NODE  = 256   # ResNet18 layer3 channels
FEATURE_CHANNEL_EDGE  = 512   # ResNet18 layer4 channels
NODE_FEATURE_DIM      = FEATURE_CHANNEL_NODE + FEATURE_CHANNEL_EDGE   # 768
GLOBAL_FEATURE_DIM    = FEATURE_CHANNEL_EDGE                          # 512
GLOBAL_STATE_DIM      = GLOBAL_FEATURE_DIM * 2                        # 1024


FIRST_ORDER = True
POSITIVE_EDGES= True
GNN_LAYER =3
# SK_TAU= 0.005
SK_TAU=0.01
SK_EMB=1
GNN_FEAT = [16, 16, 16]
GNN_LAYER = 3
EDGE_EMB=False
BATCH_SIZE=8

UNIV_SIZE=450
SK_ITER_NUM=10
SK_EPSILON=1e-10
K_FACTOR=50.
DUSTBIN_LOSS_WEIGHT=1.0




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

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 0)

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
        k_gate_enable=False,
        k_gate_thresh=0.2,
        k_match_rounding="floor",
        auth_gate_enable=False,
        auth_gate_thresh=0.5,
        k_pred_ramp=1.0,
    ):
        super(Net, self).__init__() # initialize the VGG16 model
        
        # --- Spline-Conv path ------------------------------------------------
        self.message_pass_node_features = SiameseSConvOnNodes(
            input_node_dim=NODE_FEATURE_DIM
        )

        self.build_edge_features_from_node_features = (
            SiameseNodeFeaturesToEdgeFeatures(
                total_num_nodes=self.message_pass_node_features.num_node_features
            )
        )

        # --- Affinity layers -------------------------------------------------
        self.global_state_dim = GLOBAL_STATE_DIM           # 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,                         # input  = 1024
            self.message_pass_node_features.num_node_features  # output = 768
        )

        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,                         # input  = 1024
            self.build_edge_features_from_node_features.num_edge_features
        )
        
        self.tau=SK_TAU 
        
        # Initialize my GNN Layer
        self.gnn_layer = GNN_LAYER
        for i in range(self.gnn_layer):
                tau = self.tau
                if i == 0:
                    gnn_layer = PYGNNLayer(1, 1,
                                            GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                            sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                else:
                    gnn_layer = PYGNNLayer(GNN_FEAT[i - 1] + SK_EMB, GNN_FEAT[i - 1],
                                            GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                            sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
    
        self.rescale = (320, 240)
        self.univ_size = UNIV_SIZE
        self.k_factor=K_FACTOR
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
        self.k_gate_enable = bool(k_gate_enable)
        self.k_gate_thresh = float(k_gate_thresh)
        self.k_match_rounding = str(k_match_rounding).strip().lower()
        self.auth_gate_enable = bool(auth_gate_enable)
        self.auth_gate_thresh = float(auth_gate_thresh)
        self.k_pred_ramp = float(k_pred_ramp)
        self.train_stage = None
        # Authentication head: uses [perm_ratio, k_prob] to output auth logit.
        self.auth_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        
        
        self.k_params_id = []
    # if self.regression:
        # Only implementing AFAU
        self.encoder_k = Encoder()
        self.k_params_id += [id(item) for item in self.encoder_k.parameters()]
        self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
        self.final_row = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.final_col = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.k_params_id += [id(item) for item in self.final_row.parameters()]
        self.k_params_id += [id(item) for item in self.final_col.parameters()]

        self.k_params = [
        {'params': self.encoder_k.parameters()},
        {'params': self.final_row.parameters()},
        {'params': self.final_col.parameters()}
        ]


    def _sinkhorn_with_marginals(self, scores, row_marginal, col_marginal, iters):
        eps = 1e-8
        log_mu = torch.log(row_marginal.clamp_min(eps))
        log_nu = torch.log(col_marginal.clamp_min(eps))
        log_p = log_sinkhorn_iterations(scores / self.tau, log_mu, log_nu, iters)
        return torch.exp(log_p)

    def _dustbin_sinkhorn(self, scores, nrows, ncols, ks):
        batch, max_n1, max_n2 = scores.shape
        out = scores.new_zeros(batch, max_n1 + 1, max_n2 + 1)
        real = scores.new_zeros(batch, max_n1, max_n2)
        for b in range(batch):
            n1 = int(nrows[b].item())
            n2 = int(ncols[b].item())
            if n1 == 0 and n2 == 0:
                continue
            logits = scores[b, :n1, :n2]
            aug = scores.new_zeros((n1 + 1, n2 + 1))
            if n1 > 0 and n2 > 0:
                aug[:n1, :n2] = logits
            aug[:n1, n2] = self.bin_score
            aug[n1, :n2] = self.bin_score
            aug[n1, n2] = self.bin_score

            k = ks[b]
            max_k = min(n1, n2)
            k = torch.clamp(k, min=0.0, max=float(max_k))
            row_m = scores.new_ones(n1 + 1)
            col_m = scores.new_ones(n2 + 1)
            row_m[-1] = n2 - k
            col_m[-1] = n1 - k
            row_m = row_m.clamp_min(0.0)
            col_m = col_m.clamp_min(0.0)

            p = self._sinkhorn_with_marginals(aug, row_m, col_m, SK_ITER_NUM)
            out[b, :n1 + 1, :n2 + 1] = p
            if n1 > 0 and n2 > 0:
                real[b, :n1, :n2] = p[:n1, :n2]
        return out, real

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
                loss += F.binary_cross_entropy(pred_col, gt_col, reduction='sum')
                count += n1
            if n2 > 0:
                pred_row = pred[b, n1, :n2].clamp(min=eps, max=1.0 - eps)
                gt_row = gt[b, n1, :n2].to(pred.dtype)
                loss += F.binary_cross_entropy(pred_row, gt_row, reduction='sum')
                count += n2
        if count > 0:
            loss = loss / count
        return loss

    def _compute_k_match(self, ks, gt_ks, min_points, no_pair_mask, training, auth_prob=None):
        k_pred_count = torch.nan_to_num(
            ks.view(-1) * min_points, nan=0.0, posinf=0.0, neginf=0.0
        )
        train_stage = getattr(self, "train_stage", None)
        use_pred_k_train = False
        if train_stage is None:
            use_pred_k_train = bool(getattr(self, "train_use_pred_k", False))
        else:
            use_pred_k_train = int(train_stage) == 4

        if training:
            k_match = k_pred_count if use_pred_k_train else gt_ks
        else:
            k_match = k_pred_count

        if no_pair_mask is not None:
            k_match = torch.where(no_pair_mask, torch.zeros_like(k_match), k_match)

        if not training:
            if self.k_gate_enable:
                k_match = torch.where(
                    ks.view(-1) < self.k_gate_thresh,
                    torch.zeros_like(k_match),
                    k_match,
                )
            if self.auth_gate_enable and auth_prob is not None:
                k_match = torch.where(
                    auth_prob.view(-1) < self.auth_gate_thresh,
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

    
    def forward(self, data_dict, regression=True):
        images = data_dict['images'] # Loaded from custom dataset
        points = data_dict['Ps'] # Pore locations
        n_points = data_dict['ns'] # number of pores
        A_src, A_tgt = data_dict['As'] # Adjacency Matrices
        graphs = data_dict['pyg_graphs'] # Generated by GMDataset
        batch_size = data_dict['gt_perm_mat'].shape[0] # Generated by GMDataset
        num_graphs = len(images) # number of fingerprints
        log_nans = not getattr(self, "_nan_debug_done", False)
        
        # if 'KGHs' in data_dict:
        #     logger.info("Data dict contains KGHs with type=%s", type(data_dict['KGHs']))
        # else:
        #     logger.warning("Data dict does not contain 'KGHs' key.")
        
        global_list = [] # List of fingerprint global features
        orig_graph_list = [] # Edge graphs with node feature embeddings
        node_feature_list = [] # node features
        
        # logger.info('%s' , data_dict['KGHs'])
        
        # print("Number of pores: ", n_points )
        
        # Loop through images, pores, number of pores, graphs  
        for image, point, num_p, graph in zip(images, points, n_points, graphs):
            
            # if a single image is being passed, unsqueeze dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Get node, edge and global features from image (VGG16)
            nodes = self.node_layers(image)
            if log_nans:
                _nan_stats("node_layers", nodes)
            # print(nodes)
            edges = self.edge_layers(nodes)
            if log_nans:
                _nan_stats("edge_layers", edges)
            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            if log_nans:
                _nan_stats("final_layers", global_list[-1])
            

            # L2 Norm
            nodes = normalize_over_channels(nodes)
            # print(nodes)
            edges = normalize_over_channels(edges)
            if log_nans:
                _nan_stats("norm_nodes", nodes)
                _nan_stats("norm_edges", edges)


            # arrange features
            U = concat_features(feature_align(nodes, point, num_p, self.rescale), num_p)
            edge_feats = concat_features(feature_align(edges, point, num_p, self.rescale), num_p)
            if log_nans:
                _nan_stats("feature_align_U", U)
                _nan_stats("feature_align_F", edge_feats)
            node_features = torch.cat((U, edge_feats), dim=1)
            if log_nans:
                _nan_stats("node_features_cat", node_features)
            pos = normalize_keypoints(point, image.shape)
            pos_emb = self.pos_mlp(pos)
            pos_emb = pos_emb.permute(0, 2, 1)
            pos_emb = concat_features(pos_emb, num_p)
            node_features = node_features + pos_emb
            node_feature_list.append(node_features.detach())
            # node_features = self.proj(node_features)
            graph.x = node_features
            
            # Apply Spline conv network for enhanced feature extraction
            graph = self.message_pass_node_features(graph)
            if log_nans:
                _nan_stats("message_pass_node_features", graph.x)
            orig_graph = self.build_edge_features_from_node_features(graph)
            if log_nans:
                edge_attr = orig_graph[0].edge_attr if isinstance(orig_graph, list) else orig_graph.edge_attr
                _nan_stats("build_edge_features_from_node_features", edge_attr)
            orig_graph_list.append(orig_graph)
            # visualize_pyg_data(orig_graph)
        
            # print(image.size())
            
            
            global_weights_list = [
            # self.proj(torch.cat([global_src, global_tgt], axis=-1)) for global_src, global_tgt in lexico_iter(global_list)
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
            
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]
        if log_nans:
            _nan_stats("global_weights_list[0]", global_weights_list[0] if global_weights_list else None)

        # for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list):
        #     for item in g_1:
        #         # logger.info("Graph1 item.x.size() = %s", item.x.size())
        #     for item in g_2:
        #         logger.info("Graph2 item.x.size() = %s", item.x.size())
            # mat1 dim 1 must match mat2 dim 0 maybe padding
         
        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        if log_nans and unary_affs_list:
            _nan_stats("unary_affs_list[0]", unary_affs_list[0][0] if unary_affs_list[0] else None)

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        if log_nans and quadratic_affs_list:
            _nan_stats("quadratic_affs_list[0]", quadratic_affs_list[0][0] if quadratic_affs_list[0] else None)

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        if log_nans:
            self._nan_debug_done = True

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []
        
        # logger.info("Running Sparse GNN")
        # Sparse implementation not implemented
        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
        #     kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
        #     Kp = torch.stack(pad_tensor(unary_affs), dim=0)
        #     Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
        #     K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
        #     if num_graphs == 2: data_dict['aff_mat'] = K

        #     if FIRST_ORDER:
        #         emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        #     else:
        #         emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        #     if POSITIVE_EDGES:
        #         A = (K > 0).to(K.dtype)
        #     else:
        #         A = (K != 0).to(K.dtype)

        #     emb_K = K.unsqueeze(-1)

        #     # NGM qap solver
        #     for i in range(self.gnn_layer):
        #         gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
        #         emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])
        
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)

            if FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(BATCH_SIZE, Kp.shape[1] * Kp.shape[2], 1, device=K_value.device)

            qap_emb = []
            for b in range(len(data_dict['KGHs_sparse'])):
                kro_G, kro_H = data_dict['KGHs_sparse'][b] if num_graphs == 2 else data_dict['KGHs_sparse']['{},{}'.format(idx1, idx2)]
                K_value, row_idx, col_idx = construct_sparse_aff_mat(quadratic_affs[b], unary_affs[b], kro_G, kro_H)

            # NGM qap solver
                tmp_emb = emb[b].unsqueeze(0)
                # if self.geometric:
                # Ensure index/value tensors have the same length when constructing
                # the sparse affinity matrix. Occasionally the returned row/col
                # indices may not perfectly match the value tensor due to
                # preprocessing irregularities.  To prevent runtime errors in
                # ``torch_sparse`` we truncate all tensors to the smallest common
                # length.
                common_len = min(row_idx.numel(), col_idx.numel(), K_value.numel())
                row = row_idx[:common_len].long()
                col = col_idx[:common_len].long()
                val = K_value[:common_len]
                adj = SparseTensor(row=row, col=col, value=val,
                                    sparse_sizes=(Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2]))
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    tmp_emb = gnn_layer(adj, tmp_emb, n_points[idx1], n_points[idx2], b)
                qap_emb.append(tmp_emb.squeeze(0))
                # else:
                # K_index = torch.cat((row_idx.unsqueeze(0), col_idx.unsqueeze(0)), dim=0).long()
                # A_value = torch.ones(K_value.shape, device=K_value.device)
                # tmp = torch.ones([Kp.shape[1] * Kp.shape[2]], device=K_value.device).unsqueeze(-1)
                # normed_A_value = 1 / torch.flatten(
                #     spmm(K_index, A_value, Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2], tmp))
                # A_index = torch.linspace(0, Kp.shape[1] * Kp.shape[2] - 1, Kp.shape[1] * Kp.shape[2]).unsqueeze(0)
                # A_index = torch.repeat_interleave(A_index, 2, dim=0).long().to(K_value.device)

                # for i in range(self.gnn_layer):
                #     gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                #     tmp_emb = gnn_layer(K_value, K_index, normed_A_value, A_index, tmp_emb, n_points[idx1], n_points[idx2], b)
                # qap_emb.append(tmp_emb.squeeze(0))
        emb = torch.stack(pad_tensor(qap_emb), dim=0)
                
        # logger.info("Final emb_K shape: %s", emb_K.shape)
        # logger.info("Final emb shape: %s", emb.shape)
        # print(emb)

        v = self.classifier(emb)
        s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

        ss_base = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

        # Calculate the minimum number of keypoints between paired images
        min_point_list = [int(min(n_points[idx1][b], n_points[idx2][b]))
                          for b in range(data_dict['gt_perm_mat'].shape[0])]

        min_point_tensor = torch.tensor(min_point_list, dtype=torch.float32,
                                        device=s.device)
        min_point_tensor_safe = torch.clamp(min_point_tensor, min=1.0)

        # Ground truth k derived from real<->real block (exclude dustbin row/col)
        gt_ks = torch.tensor([
            torch.sum(
                data_dict['gt_perm_mat'][i, :int(n_points[idx1][i].item()), :int(n_points[idx2][i].item())]
            )
            for i in range(data_dict['gt_perm_mat'].shape[0])
        ], dtype=torch.float32, device=s.device)

        # If it is an imposter match, enforce k = 0 (no true correspondences)
        no_pair_mask = None
        if 'label' in data_dict:
            labels = data_dict['label'].to(s.device).view(-1).long()
            imposter_mask = labels == 0
            if imposter_mask.any():
                gt_ks = gt_ks.clone()
                gt_ks[imposter_mask] = 0.0
            no_pair_mask = imposter_mask | (gt_ks == 0)
        else:
            no_pair_mask = (gt_ks == 0)

        if self.regression:
            print("Predicting K using AFAU")
            dummy_row = self.univ_size - s.shape[1]
            dummy_col = self.univ_size - s.shape[2]
            assert dummy_row >= 0 and dummy_col >= 0
            
            # AFAU
            init_row_emb = torch.zeros((batch_size, int(torch.max(n_points[idx1])), self.univ_size), dtype=torch.float32, device=s.device)

            init_col_emb = torch.zeros((batch_size, int(torch.max(n_points[idx2])), self.univ_size), dtype=torch.float32, device=s.device)

            for b in range(batch_size):
                index = torch.linspace(0, n_points[idx2][b].item() - 1, n_points[idx2][b].item(), dtype=torch.long, device=s.device).unsqueeze(1)
                init_col_emb_one = torch.zeros(int(torch.max(n_points[idx2])), self.univ_size, dtype=torch.float32, device=s.device).scatter_(1, index, 1)
                init_col_emb[b] = init_col_emb_one

            out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss_base.detach())
            # out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss)

            out_emb_row = torch.nn.functional.pad(out_emb_row, (0, 0, 0, dummy_row), value=float('-inf')).permute(0, 2, 1)
            out_emb_col = torch.nn.functional.pad(out_emb_col, (0, 0, 0, dummy_col), value=float('-inf')).permute(0, 2, 1)
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
            ks = torch.where(min_point_tensor > 0, gt_ks / min_point_tensor_safe, torch.zeros_like(gt_ks))

        k_match, k_pred_count = self._compute_k_match(
            ks,
            gt_ks,
            min_point_tensor,
            no_pair_mask,
            training=self.training,
            auth_prob=None,
        )

        def _build_matching(k_match_local):
            dustbin_local, real_local = self._dustbin_sinkhorn(
                s, n_points[idx1], n_points[idx2], k_match_local
            )
            _, topk_mask_local = soft_topk(
                real_local,
                k_match_local.view(-1),
                SK_ITER_NUM,
                self.tau,
                n_points[idx1],
                n_points[idx2],
                True,
            )
            refined_local = topk_mask_local * real_local
            ds_local = dustbin_local.clone()
            ds_local[:, :real_local.shape[1], :real_local.shape[2]] = refined_local

            x_local = hungarian(refined_local, n_points[idx1], n_points[idx2])
            top_indices_local = torch.argsort(
                x_local.mul(refined_local).reshape(x_local.shape[0], -1), descending=True, dim=-1
            )
            x_local = torch.zeros_like(refined_local)
            x_local = greedy_perm(x_local, top_indices_local, k_match_local)

            perm_full_local = refined_local.new_zeros(
                refined_local.shape[0], refined_local.shape[1] + 1, refined_local.shape[2] + 1
            )
            perm_full_local[:, :refined_local.shape[1], :refined_local.shape[2]] = x_local
            for b in range(perm_full_local.shape[0]):
                n1 = int(n_points[idx1][b].item())
                n2 = int(n_points[idx2][b].item())
                if n1 > 0:
                    row_sums = x_local[b, :n1, :n2].sum(dim=1)
                    perm_full_local[b, :n1, n2] = (row_sums == 0).to(perm_full_local.dtype)
                if n2 > 0:
                    col_sums = x_local[b, :n1, :n2].sum(dim=0)
                    perm_full_local[b, n1, :n2] = (col_sums == 0).to(perm_full_local.dtype)
                perm_full_local[b, n1, n2] = 0.0

            reject_rows_local = []
            reject_cols_local = []
            if getattr(self, "dustbin_reject_enable", False):
                margin = float(getattr(self, "dustbin_reject_margin", 0.0))
                for b in range(perm_full_local.shape[0]):
                    n1 = int(n_points[idx1][b].item())
                    n2 = int(n_points[idx2][b].item())
                    p = dustbin_local[b]
                    row_reject = p.new_zeros(n1, dtype=torch.bool)
                    col_reject = p.new_zeros(n2, dtype=torch.bool)
                    if n1 > 0 and n2 > 0:
                        best_j = refined_local[b, :n1, :n2].argmax(dim=1)
                        best_p = p[torch.arange(n1, device=p.device), best_j]
                        dust_p = p[:n1, n2]
                        row_reject = dust_p >= (best_p + margin)
                        best_i = refined_local[b, :n1, :n2].argmax(dim=0)
                        best_p_col = p[best_i, torch.arange(n2, device=p.device)]
                        dust_p_col = p[n1, :n2]
                        col_reject = dust_p_col >= (best_p_col + margin)
                    if n1 > 0:
                        x_local[b, :n1, :n2][row_reject] = 0.0
                    if n2 > 0:
                        x_local[b, :n1, :n2][:, col_reject] = 0.0
                    if n1 > 0:
                        row_sums = x_local[b, :n1, :n2].sum(dim=1)
                        perm_full_local[b, :n1, n2] = (row_sums == 0).to(perm_full_local.dtype)
                    if n2 > 0:
                        col_sums = x_local[b, :n1, :n2].sum(dim=0)
                        perm_full_local[b, n1, :n2] = (col_sums == 0).to(perm_full_local.dtype)
                    perm_full_local[b, :n1, :n2] = x_local[b, :n1, :n2]
                    reject_rows_local.append(row_reject)
                    reject_cols_local.append(col_reject)
            else:
                for b in range(perm_full_local.shape[0]):
                    n1 = int(n_points[idx1][b].item())
                    n2 = int(n_points[idx2][b].item())
                    reject_rows_local.append(dustbin_local.new_zeros(n1, dtype=torch.bool))
                    reject_cols_local.append(dustbin_local.new_zeros(n2, dtype=torch.bool))

            return ds_local, x_local, perm_full_local, reject_rows_local, reject_cols_local, dustbin_local

        ds_mat, x, perm_full, reject_rows, reject_cols, dustbin_mat = _build_matching(k_match)

        supervised_ks = torch.where(min_point_tensor > 0, gt_ks / min_point_tensor_safe, torch.zeros_like(gt_ks))
        if no_pair_mask is not None:
            supervised_ks = torch.where(no_pair_mask, torch.zeros_like(supervised_ks), supervised_ks)

        # Authentication features and logits.
        perm_sum = x.sum(dim=(1, 2)).float()
        perm_ratio = torch.where(
            min_point_tensor_safe > 0,
            perm_sum / min_point_tensor_safe,
            torch.zeros_like(perm_sum),
        )
        auth_input = torch.stack([perm_ratio, ks.view(-1)], dim=1)
        auth_logit = self.auth_head(auth_input).squeeze(-1)
        auth_prob = torch.sigmoid(auth_logit)

        if (not self.training) and self.auth_gate_enable:
            if (auth_prob.view(-1) < self.auth_gate_thresh).any():
                k_match, k_pred_count = self._compute_k_match(
                    ks,
                    gt_ks,
                    min_point_tensor,
                    no_pair_mask,
                    training=False,
                    auth_prob=auth_prob,
                )
                ds_mat, x, perm_full, reject_rows, reject_cols, dustbin_mat = _build_matching(k_match)
        k_reg_loss = ds_mat.new_tensor(0.0)
        k_cls_loss = ds_mat.new_tensor(0.0)
        k_count_loss = ds_mat.new_tensor(0.0)
        if self.regression:
            k_reg_loss = torch.nn.functional.mse_loss(ks, supervised_ks) * self.k_factor
            if "label" in data_dict:
                labels = data_dict["label"].to(ds_mat.device).view(-1).float()
                k_cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    k_logits.view(-1), labels
                )
            count_err = torch.nn.functional.smooth_l1_loss(
                k_pred_count, gt_ks, reduction="none"
            )
            count_err = count_err / min_point_tensor_safe
            k_count_loss = count_err.mean()
            ks_loss = (self.k_reg_weight * k_reg_loss) + (self.k_cls_weight * k_cls_loss) + k_count_loss
            ks_error = torch.nn.functional.l1_loss(ks * min_point_tensor, gt_ks)
        else:
            ks_loss = 0.0
            ks_error = 0.0

        if 'gt_perm_mat' in data_dict:
            dustbin_loss = self._dustbin_supervision_loss(
                ds_mat, data_dict['gt_perm_mat'], n_points[idx1], n_points[idx2]
            ) * self.dustbin_loss_weight
            eps = 1e-8
            sg_loss = ds_mat.new_tensor(0.0)
            sg_count = ds_mat.new_tensor(0.0)
            for b in range(dustbin_mat.shape[0]):
                n1 = int(n_points[idx1][b].item())
                n2 = int(n_points[idx2][b].item())
                if n1 == 0 and n2 == 0:
                    continue
                gt_block = data_dict['gt_perm_mat'][b, :n1, :n2]
                p = dustbin_mat[b]
                if n1 > 0 and n2 > 0:
                    match_mask = gt_block > 0.5
                    if match_mask.any():
                        sg_loss -= torch.log(p[:n1, :n2].clamp(min=eps))[match_mask].sum()
                        sg_count += match_mask.sum()
                if n1 > 0:
                    row_unmatched = gt_block.sum(dim=1) == 0 if n2 > 0 else torch.ones(n1, dtype=torch.bool, device=p.device)
                    if row_unmatched.any():
                        sg_loss -= torch.log(p[:n1, n2].clamp(min=eps))[row_unmatched].sum()
                        sg_count += row_unmatched.sum()
                if n2 > 0:
                    col_unmatched = gt_block.sum(dim=0) == 0 if n1 > 0 else torch.ones(n2, dtype=torch.bool, device=p.device)
                    if col_unmatched.any():
                        sg_loss -= torch.log(p[n1, :n2].clamp(min=eps))[col_unmatched].sum()
                        sg_count += col_unmatched.sum()
            sg_dustbin_loss = sg_loss / sg_count if sg_count > 0 else ds_mat.new_tensor(0.0)
        else:
            dustbin_loss = 0.0
            sg_dustbin_loss = 0.0

        s_list.append(ds_mat)
        x_list.append(perm_full)
        indices.append((idx1, idx2))
        
        # print(x)
        # print(ss_out)

        
        data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0],
                'ks_loss': ks_loss,
                'ks_error': ks_error,
                'dustbin_loss': dustbin_loss,
                'sg_dustbin_loss': sg_dustbin_loss,
                'gt_ks': gt_ks,
                'k_pred_ratio': ks,
                'k_pred_count': k_pred_count,
                'k_match_count': k_match,
                'k_count_loss': k_count_loss,
                'k_prob': ks,
                'k_logit': k_logits,
                'k_reg_loss': k_reg_loss,
                'k_cls_loss': k_cls_loss,
                'auth_logit': auth_logit,
                'auth_prob': auth_prob,
                'rejected_rows': reject_rows,
                'rejected_cols': reject_cols,
                'ns': [n_points[idx1] + 1, n_points[idx2] + 1],
                'has_dustbin': True,
            })
        

        
        return data_dict
