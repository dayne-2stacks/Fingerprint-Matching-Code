import logging
from feature_extractor import VGG16_bn_final as CNN
from spline_conv import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from feature_align import feature_align
from affinity_layer import InnerProductWithWeightsAffinity
import torch
import torch.nn as nn
from utils.pad_tensor import pad_tensor
from utils.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from gnn import PYGNNLayer
from sinkhorn import Sinkhorn
from soft_topk import soft_topk, greedy_perm
from hungarian import hungarian
from afau import Encoder

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
FEATURE_CHANNEL = 512

FIRST_ORDER = True
POSITIVE_EDGES= True
GNN_LAYER =3
# SK_TAU= 0.005
SK_TAU=0.01
SK_EMB=1
GNN_FEAT = [16, 16, 16]
GNN_LAYER = 3
EDGE_EMB=False
BATCH_SIZE=12

UNIV_SIZE=400
SK_ITER_NUM=15
SK_EPSILON=1e-10
K_FACTOR=5.




# Return as iterable combinations
def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


# CNN is the VGG16 feature extractor with final fully connected layers

# Contains methods {
#     node_layers - features for nodes (pores)
#     edge_layers - features for edges
#     final_layers - global features of larger fingerprint
# }

class Net(CNN):
    def __init__(self, regression=False):
        super(Net, self).__init__() # initialize the VGG16 model
        
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        logger.info("Initialized Net with message_pass_node_features.num_node_features=%d",
                self.message_pass_node_features.num_node_features)
        
        self.global_state_dim = FEATURE_CHANNEL * 2
        # Model to create affinity matrices
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        
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
        
        # Classify fingerprint
        self.classifier = nn.Linear(GNN_FEAT[-1] + SK_EMB, 1)
        
        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, tau=self.tau, epsilon=SK_EPSILON)
        self.regression = regression
        self.mean_k = True
        self.trainings=True
        
        self.k_params_id = []
    # if self.regression:
        # Only implementing AFAU
        self.encoder_k = Encoder()
        self.k_params_id += [id(item) for item in self.encoder_k.parameters()]
        self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
        self.final_row = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.final_col = nn.Sequential(
            nn.Linear(self.univ_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.k_params_id += [id(item) for item in self.final_row.parameters()]
        self.k_params_id += [id(item) for item in self.final_col.parameters()]

        self.k_params = [
        {'params': self.encoder_k.parameters()},
        {'params': self.final_row.parameters()},
        {'params': self.final_col.parameters()}
        ]

    def forward(self, data_dict, regression=True):
        images = data_dict['images'] # Loaded from custom dataset
        points = data_dict['Ps'] # Pore locations
        n_points = data_dict['ns'] # number of pores
        A_src, A_tgt = data_dict['As'] # Adjacency Matrices
        graphs = data_dict['pyg_graphs'] # Generated by GMDataset
        batch_size = data_dict['gt_perm_mat'].shape[0] # Generated by GMDataset
        num_graphs = len(images) # number of fingerprints
        
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
            # print(nodes)
            edges = self.edge_layers(nodes)
            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            
            # L2 Norm
            nodes = normalize_over_channels(nodes)
            # print(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, point, num_p, self.rescale), num_p)
            F = concat_features(feature_align(edges, point, num_p, self.rescale), num_p)
            node_features = torch.cat((U, F), dim=1)
            # This detaches the node features that would cause no gradients to flow back into the model.
            node_feature_list.append(node_features)
            # node_features = self.proj(node_features)
            graph.x = node_features
            
            # Apply Spline conv network for enhanced feature extraction
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)
            # visualize_pyg_data(orig_graph)
        
            # print(image.size())
            
            
            global_weights_list = [
            # self.proj(torch.cat([global_src, global_tgt], axis=-1)) for global_src, global_tgt in lexico_iter(global_list)
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
            
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

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

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

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
                adj = SparseTensor(row=row_idx.long(), col=col_idx.long(), value=K_value,
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

        ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

        gt_ks = torch.tensor(
            [torch.sum(data_dict['gt_perm_mat'][i]) for i in range(data_dict['gt_perm_mat'].shape[0])],
            dtype=torch.float32, device=s.device)

        min_point_list = [int(min(n_points[0][b], n_points[1][b])) for b in range(data_dict['gt_perm_mat'].shape[0])]

        min_point_tensor = torch.tensor(min_point_list, dtype=torch.float32, device=s.device)

        if self.regression:
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

            out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss.detach())
            out_emb_row = torch.nn.functional.pad(out_emb_row, (0, 0, 0, dummy_row), value=float('-inf')).permute(0, 2, 1)
            out_emb_col = torch.nn.functional.pad(out_emb_col, (0, 0, 0, dummy_col), value=float('-inf')).permute(0, 2, 1)
            global_row_emb = self.maxpool(out_emb_row).squeeze(-1)
            global_col_emb = self.maxpool(out_emb_col).squeeze(-1)
            k_row = self.final_row(global_row_emb).squeeze(-1)
            k_col = self.final_col(global_col_emb).squeeze(-1)
            if self.mean_k:
                ks = (k_row + k_col) / 2
            else:
                ks = k_row
            
            
        else:
            ks = gt_ks / min_point_tensor 
            
        if self.trainings:
            _, ss_out = soft_topk(ss, gt_ks.view(-1), SK_ITER_NUM, self.tau, n_points[idx1], n_points[idx2],
                                True)
        else:
            _, ss_out = soft_topk(ss, ks.view(-1) * min_point_tensor, SK_ITER_NUM, self.tau, n_points[idx1],
                                    n_points[idx2], True)

        supervised_ks = gt_ks / min_point_tensor
        #print("Groundtruth:", gt_ks)
        #print("Supervised", supervised_ks)

        if self.regression:
            ks_loss = torch.nn.functional.mse_loss(ks, supervised_ks) * self.k_factor
            ks_error = torch.nn.functional.l1_loss(ks * min_point_tensor, gt_ks)
        else:
            ks_loss = 0.
            ks_error = 0.

        x = hungarian(ss_out, n_points[idx1], n_points[idx2])
        top_indices = torch.argsort(x.mul(ss_out).reshape(x.shape[0], -1), descending=True, dim=-1)
        x = torch.zeros(ss_out.shape, device=ss_out.device)
        x = greedy_perm(x, top_indices, ks.view(-1) * min_point_tensor)
        s_list.append(ss_out)
        x_list.append(x)
        indices.append((idx1, idx2))  
        
        # print(x)
        # print(ss_out)

        
        data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0],
                'ks_loss': ks_loss,
                'ks_error': ks_error
            })
        
        return data_dict
