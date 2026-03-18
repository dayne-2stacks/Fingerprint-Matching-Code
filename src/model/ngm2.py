import itertools
from torch_sparse import spmm, SparseTensor

import torch
import torch.nn as nn


from src.model.affinity_layer import InnerProductWithWeightsAffinity
from src.model.spline_conv import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.feature_align import feature_align
from utils.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from utils.pad_tensor import pad_tensor
from src.model.gnn import GNNLayer, SPGNNLayer, PYGNNLayer
# from src.gconv import Siamese_Gconv
# from models.PCA.affinity_layer import Affinity
from src.model.afau import Encoder, TensorNetworkModule, DenseAttentionModule
from src.model.sinkhorn import Sinkhorn
from src.model.soft_topk import soft_topk, greedy_perm
from utils.hungarian import hungarian
from src.model.feature_extractor import ResNet34_base as CNN


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
BATCH_SIZE = 8

UNIV_SIZE = 600
SK_ITER_NUM = 25
SK_EPSILON = 1e-10
K_FACTOR = 5.0
RESCALE = (320, 240)
CROPSIZE = (240, 240)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=NODE_FEATURE_DIM)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = GLOBAL_STATE_DIM
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        # self.trainings = True
        self.sparse = True
        self.rescale = RESCALE
        self.tau = SK_TAU
        self.univ_size = UNIV_SIZE
        self.gnn_layer = GNN_LAYER

        self.k_factor = K_FACTOR
        self.reg_hidden_feat = 8
        # self.regression = cfg.AFA.REGRESSION
        self.afau = True
        self.mean_k = True

        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, tau=self.tau, epsilon=SK_EPSILON)

        if not self.sparse:
            for i in range(self.gnn_layer):
                tau = SK_TAU
                if i == 0:
                    gnn_layer = GNNLayer(1, 1,
                                         GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                         sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                else:
                    gnn_layer = GNNLayer(GNN_FEAT[i - 1] + SK_EMB, GNN_FEAT[i - 1],
                                         GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                         sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        else:
            self.geometric = True
            if self.geometric:
                for i in range(self.gnn_layer):
                    tau = SK_TAU
                    if i == 0:
                        gnn_layer = PYGNNLayer(1, 1,
                                               GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                               sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                    else:
                        gnn_layer = PYGNNLayer(GNN_FEAT[i - 1] + SK_EMB, GNN_FEAT[i - 1],
                                               GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                               sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                    self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            else:
                for i in range(self.gnn_layer):
                    tau = SK_TAU
                    if i == 0:
                        gnn_layer = SPGNNLayer(1, 1,
                                               GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                               sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                    else:
                        gnn_layer = SPGNNLayer(GNN_FEAT[i - 1] + SK_EMB, GNN_FEAT[i - 1],
                                               GNN_FEAT[i] + SK_EMB, GNN_FEAT[i],
                                               sk_channel=SK_EMB, sk_tau=tau, edge_emb=EDGE_EMB)
                    self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(GNN_FEAT[-1] + SK_EMB, 1)
        self.regression = True
        if self.regression:
            self.k_params_id = []
            if self.afau:
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
            # else:
            #     self.k_params = []
            #     for i in range(self.k_gnn_layer):
            #         if i == 0:
            #             gnn_layer_k = Siamese_Gconv(FEATURE_CHANNEL * 2, cfg.AFA.REG_HIDDEN_FEAT)
            #         else:
            #             gnn_layer_k = Siamese_Gconv(cfg.AFA.REG_HIDDEN_FEAT, cfg.AFA.REG_HIDDEN_FEAT)
            #         self.add_module('_k_gnn_layer_{}'.format(i), gnn_layer_k)
            #         self.add_module('_affinity_{}'.format(i), Affinity(cfg.AFA.REG_HIDDEN_FEAT))

            #         self.k_params_id += [id(item) for item in eval('self._k_gnn_layer_{}'.format(i)).parameters()]
            #         self.k_params_id += [id(item) for item in eval('self._affinity_{}'.format(i)).parameters()]
            #         self.k_params.append({'params': eval('self._k_gnn_layer_{}'.format(i)).parameters()})
            #         self.k_params.append({'params': eval('self._affinity_{}'.format(i)).parameters()})

            #         if i == self.k_gnn_layer - 2:  # only second last layer will have cross-graph module
            #             self.add_module('_cross_graph_{}'.format(i), nn.Linear(cfg.AFA.REG_HIDDEN_FEAT * 2, cfg.AFA.REG_HIDDEN_FEAT))
            #             self.k_params_id += [id(item) for item in eval('self._cross_graph_{}'.format(i)).parameters()]
            #             self.k_params.append({'params': eval('self._cross_graph_{}'.format(i)).parameters()})

                # self._attn_pool_1 = DenseAttentionModule(8)
                # self._attn_pool_2 = DenseAttentionModule(8)

                # self.k_params_id += [id(item) for item in self._attn_pool_1.parameters()]
                # self.k_params_id += [id(item) for item in self._attn_pool_2.parameters()]
                # self.k_params.append({'params': self._attn_pool_1.parameters()})
                # self.k_params.append({'params': self._attn_pool_2.parameters()})

                # self._ntnet = TensorNetworkModule(filters=8, tensor_neurons=16)

                # self.k_params_id += [id(item) for item in self._ntnet.parameters()]
                # self.k_params.append({'params': self._ntnet.parameters()})

                # self._final_reg = nn.Sequential(
                #     nn.Linear(16, 8),
                #     nn.ReLU(),
                #     nn.Linear(8, 1),
                #     nn.Sigmoid()
                # )

                # self.k_params_id += [id(item) for item in self._final_reg.parameters()]
                # self.k_params.append({'params': self._final_reg.parameters()})


    def forward(
        self,
        data_dict,
        regression
    ):
        self.trainings = self.training
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        A_src, A_tgt = data_dict['As']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['gt_perm_mat'].shape[0]
        num_graphs = len(images)

        global_list = []
        orig_graph_list = []
        node_feature_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            node_feature_list.append(node_features.detach())
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

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

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            if not self.sparse:
                kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
                if num_graphs == 2: data_dict['aff_mat'] = K

                if FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

                if POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                emb_K = K.unsqueeze(-1)

                # NGM qap solver
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])
            else:
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
                    if self.geometric:
                        adj = SparseTensor(row=row_idx.long(), col=col_idx.long(), value=K_value,
                                           sparse_sizes=(Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2]))
                        for i in range(self.gnn_layer):
                            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                            tmp_emb = gnn_layer(adj, tmp_emb, n_points[idx1], n_points[idx2], b)
                        qap_emb.append(tmp_emb.squeeze(0))
                    else:
                        K_index = torch.cat((row_idx.unsqueeze(0), col_idx.unsqueeze(0)), dim=0).long()
                        A_value = torch.ones(K_value.shape, device=K_value.device)
                        tmp = torch.ones([Kp.shape[1] * Kp.shape[2]], device=K_value.device).unsqueeze(-1)
                        normed_A_value = 1 / torch.flatten(
                            spmm(K_index, A_value, Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2], tmp))
                        A_index = torch.linspace(0, Kp.shape[1] * Kp.shape[2] - 1, Kp.shape[1] * Kp.shape[2]).unsqueeze(0)
                        A_index = torch.repeat_interleave(A_index, 2, dim=0).long().to(K_value.device)

                        for i in range(self.gnn_layer):
                            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                            tmp_emb = gnn_layer(K_value, K_index, normed_A_value, A_index, tmp_emb, n_points[idx1], n_points[idx2], b)
                        qap_emb.append(tmp_emb.squeeze(0))
                emb = torch.stack(pad_tensor(qap_emb), dim=0)

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

            gt_ks = torch.tensor(
                [torch.sum(data_dict['gt_perm_mat'][i]) for i in range(data_dict['gt_perm_mat'].shape[0])],
                dtype=torch.float32, device=s.device)

            min_point_list = [int(min(n_points[0][b], n_points[1][b])) for b in range(data_dict['gt_perm_mat'].shape[0])]

            min_point_tensor = torch.tensor(min_point_list, dtype=torch.float32, device=s.device)
            no_pair_mask = None
            if data_dict.get("label") is not None:
                labels = torch.as_tensor(data_dict["label"], device=s.device).view(-1).long()
                no_pair_mask = labels == 0
                if no_pair_mask.any():
                    gt_ks = gt_ks.clone()
                    gt_ks[no_pair_mask] = 0.0

            

            if regression:
                dummy_row = self.univ_size - s.shape[1]
                dummy_col = self.univ_size - s.shape[2]
                assert dummy_row >= 0 and dummy_col >= 0

                # if not self.afau:
                #     emb1 = torch.zeros((batch_size, int(torch.max(n_points[idx1])), 2 * FEATURE_CHANNEL_NODE),
                #                                dtype=torch.float32, device=s.device)

                #     emb2 = torch.zeros((batch_size, int(torch.max(n_points[idx2])), 2 * FEATURE_CHANNEL_NODE),
                #                                dtype=torch.float32, device=s.device)

                #     mask1 = torch.zeros((batch_size, int(torch.max(n_points[idx1]))),
                #                                dtype=torch.float32, device=s.device)
                #     mask2 = torch.zeros((batch_size, int(torch.max(n_points[idx2]))),
                #                                dtype=torch.float32, device=s.device)

                #     total_nodes_1 = 0
                #     total_nodes_2 = 0
                #     for b in range(batch_size):
                #         emb1_one = node_feature_list[0][total_nodes_1: total_nodes_1 + n_points[idx1][b]]
                #         emb1[b, 0: n_points[idx1][b]] = emb1_one
                #         emb2_one = node_feature_list[1][total_nodes_2: total_nodes_2 + n_points[idx2][b]]
                #         emb2[b, 0: n_points[idx2][b]] = emb2_one
                #         total_nodes_1 += n_points[idx1][b]
                #         total_nodes_2 += n_points[idx2][b]
                #         mask1_one = torch.ones((n_points[idx1][b]),
                #                                dtype=torch.float32, device=s.device)
                #         mask2_one = torch.ones((n_points[idx2][b]),
                #                                dtype=torch.float32, device=s.device)
                #         mask1[b, 0: n_points[idx1][b]] = mask1_one
                #         mask2[b, 0: n_points[idx2][b]] = mask2_one

                #     for i in range(self.k_gnn_layer):
                #         gnn_layer = getattr(self, '_k_gnn_layer_{}'.format(i))
                #         emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                #         affinity = getattr(self, '_affinity_{}'.format(i))
                #         s = affinity(emb1, emb2)
                #         s = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

                #         if i == self.k_gnn_layer - 2:
                #             cross_graph = getattr(self, '_cross_graph_{}'.format(i))
                #             new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                #             new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                #             emb1 = new_emb1
                #             emb2 = new_emb2

                #     global_emb1 = self._attn_pool_1(emb1, mask1)
                #     global_emb2 = self._attn_pool_2(emb2, mask2)

                #     sim = self._ntnet(global_emb1, global_emb2)
                #     ks = self._final_reg(sim).squeeze(-1)

                # else:
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
                ks = torch.where(
                    min_point_tensor > 0,
                    gt_ks / min_point_tensor,
                    torch.zeros_like(gt_ks),
                )

            if self.trainings:
                _, ss_out = soft_topk(ss, gt_ks.view(-1), SK_ITER_NUM, self.tau, n_points[idx1], n_points[idx2],
                                  True)
            else:
                _, ss_out = soft_topk(ss, ks.view(-1) * min_point_tensor, SK_ITER_NUM, self.tau, n_points[idx1],
                                      n_points[idx2], True)

            supervised_ks = gt_ks / min_point_tensor
             


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

        print("GT KS:", gt_ks)
    
        data_dict.update({
            'ds_mat': s_list[0],
            'perm_mat': x_list[0],
            'ks_loss': ks_loss,
            'ks_error': ks_error,
            'gt_ks': gt_ks,
            "ns": [n_points[idx1], n_points[idx2]]
            
        })

        return data_dict
