import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
import random
from utils.build_graphs import build_graphs
from utils.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d



RESCALE=(320, 240)
SRC_GRAPH_CONSTRUCT="tri"
TGT_GRAPH_CONSTRUCT="same"
SYM_ADJACENCY=True
NORM_MEANS= [0.485, 0.456, 0.406] 
NORM_STD=[0.229, 0.224, 0.225]
MAX_PROB_SIZE=-1

TYPE = '2GM'
FP16 = False
RANDOM_SEED=123
BATCH_SIZE=16
DATALOADER_NUM=6


class GMDataset(Dataset):
    def __init__(self, name, bm, length, using_all_graphs=False, cls=None, problem='2GM'):
        self.name = name
        self.bm = bm
        self.using_all_graphs = using_all_graphs
        self.obj_size = self.bm.obj_resize
        self.test = True if self.bm.sets == 'test' else False
        self.cls = None if cls in ['none', 'all'] else cls

        if self.cls is None:
            self.classes = self.bm.classes
        else:
            self.classes = [self.cls]

        self.problem_type = problem
        self.img_num_list = self.bm.compute_img_num(self.classes[0])

     
        self.id_combination, self.length = self.bm.get_id_combination(self.cls)
        self.length_list = []
        for cls in self.classes:
            cls_length = self.bm.compute_length(cls)
            self.length_list.append(cls_length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == '2GM':
            return self.get_pair(idx, self.cls)
        else:
            raise NameError("Unknown problem type: {}".format(self.problem_type))

    @staticmethod
    def to_pyg_graph(A, P):
        rescale = max(RESCALE)

        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5  # from Rolink's paper
        edge_index = np.nonzero(A)
        edge_attr = edge_feat[edge_index]

        edge_attr = np.clip(edge_attr, 0, 1)
        assert (edge_attr > -1e-5).all(), P

        o3_A = np.expand_dims(A, axis=0) * np.expand_dims(A, axis=1) * np.expand_dims(A, axis=2)
        hyperedge_index = np.nonzero(o3_A)

        pyg_graph = pyg.data.Data(
            x=torch.tensor(P / rescale).to(torch.float32),
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr).to(torch.float32),
            hyperedge_index=torch.tensor(np.array(hyperedge_index), dtype=torch.long),
        )
        return pyg_graph

    def get_pair(self, idx, cls):
        #anno_pair, perm_mat = self.bm.get_pair(self.cls if self.cls is not None else
        #                                       (idx % (BATCH_SIZE * len(self.classes))) // BATCH_SIZE)
        cls_num = random.randrange(0, len(self.classes))
        ids = list(self.id_combination[cls_num][idx % self.length_list[cls_num]])
        anno_pair, perm_mat_, id_list = self.bm.get_data(ids)
        perm_mat = perm_mat_[(0, 1)].toarray()
        while min(perm_mat.shape[0], perm_mat.shape[1]) <= 2 or perm_mat.size >= MAX_PROB_SIZE > 0 or perm_mat.sum() == 0:
            anno_pair, perm_mat_, id_list = self.bm.rand_get_data(cls)
            perm_mat = perm_mat_[(0, 1)].toarray()

        cls = [anno['cls'] for anno in anno_pair]
        P1 = [(kp['x'], kp['y']) for kp in anno_pair[0]['kpts']]
        P2 = [(kp['x'], kp['y']) for kp in anno_pair[1]['kpts']]

        n1, n2 = len(P1), len(P2)
        univ_size = [anno['univ_size'] for anno in anno_pair]

        P1 = np.array(P1)
        P2 = np.array(P2)

        A1, G1, H1, e1 = build_graphs(P1, n1, stg=SRC_GRAPH_CONSTRUCT, sym=SYM_ADJACENCY)
        if TGT_GRAPH_CONSTRUCT == 'same':
            G2 = perm_mat.transpose().dot(G1)
            H2 = perm_mat.transpose().dot(H1)
            A2 = G2.dot(H2.transpose())
            e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(P2, n2, stg=TGT_GRAPH_CONSTRUCT, sym=SYM_ADJACENCY)

        pyg_graph1 = self.to_pyg_graph(A1, P1)
        pyg_graph2 = self.to_pyg_graph(A2, P2)

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1, P2]],
                    'ns': [torch.tensor(x) for x in [n1, n2]],
                    'es': [torch.tensor(x) for x in [e1, e2]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1, G2]],
                    'Hs': [torch.Tensor(x) for x in [H1, H2]],
                    'As': [torch.Tensor(x) for x in [A1, A2]],
                    'pyg_graphs': [pyg_graph1, pyg_graph2],
                    'cls': [str(x) for x in cls],
                    'id_list': id_list,
                    'univ_size': [torch.tensor(int(x)) for x in univ_size],
                    }

        imgs = [anno['img'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(NORM_MEANS,NORM_STD)
                    ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['kpts'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['kpts']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['kpts']], axis=-1)
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict


class QAPDataset(Dataset):
    def __init__(self, name, length, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args, cls=cls)
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.length = length

    def __len__(self):
        #return len(self.ds.data_list)
        return self.length

    def __getitem__(self, idx):
        Fi, Fj, perm_mat, sol, name = self.ds.get_pair(idx % len(self.ds.data_list))
        if perm_mat.size <= 2 * 2 or perm_mat.size >=MAX_PROB_SIZE > 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        #if np.max(ori_aff_mat) > 0:
        #    norm_aff_mat = ori_aff_mat / np.mean(ori_aff_mat)
        #else:
        #    norm_aff_mat = ori_aff_mat

        ret_dict = {'Fi': Fi,
                    'Fj': Fj,
                    'gt_perm_mat': perm_mat,
                    'ns': [torch.tensor(x) for x in perm_mat.shape],
                    'solution': torch.tensor(sol),
                    'name': name,
                    'univ_size': [torch.tensor(x) for x in perm_mat.shape],}

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == pyg.data.Data:
            ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == tuple:
            ret = inp

        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive Kronecker product here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if TYPE == '2GM' and len(ret['Gs']) == 2 and len(ret['Hs']) == 2:
            G1, G2 = ret['Gs']
            H1, H2 = ret['Hs']
            if FP16:
                sparse_dtype = np.float16
            else:
                sparse_dtype = np.float32
            if G1.shape[0] > 1:
                KGHs_sparse = []
                for b in range(G1.shape[0]):
                    K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in
                           zip(G2[b].unsqueeze(0), G1[b].unsqueeze(0))]  # 1 as source graph, 2 as target graph
                    K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2[b].unsqueeze(0), H1[b].unsqueeze(0))]

                    # if 'NGM' in cfg and NGM.SPARSE_MODEL:
                    K1G_sparse = CSCMatrix3d(K1G)
                    K1H_sparse = CSCMatrix3d(K1H).transpose()
                    KGHs_sparse.append((K1G_sparse.indices, K1H_sparse.indices))
                ret['KGHs_sparse'] = KGHs_sparse
            else:
                K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]  # 1 as source graph, 2 as target graph
                K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]

                # if 'NGM' in cfg and NGM.SPARSE_MODEL:
                K1G_sparse = CSCMatrix3d(K1G)
                K1H_sparse = CSCMatrix3d(K1H).transpose()
                ret['KGHs_sparse'] = [(K1G_sparse.indices, K1H_sparse.indices)]
            # else:
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in
                   zip(G2, G1)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]

            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()
            ret['KGHs'] = K1G, K1H
        else:
            raise ValueError('Data type not understood.')

    if 'Fi' in ret and 'Fj' in ret:
        Fi = ret['Fi']
        Fj = ret['Fj']
        aff_mat = kronecker_torch(Fj, Fi)
        ret['aff_mat'] = aff_mat

    ret['batch_size'] = len(data)
    ret['univ_size'] = torch.tensor([max(*[item[b] for item in ret['univ_size']]) for b in range(ret['batch_size'])])

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(RANDOM_SEED + worker_id)
    np.random.seed(RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
