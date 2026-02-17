import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
import random
import re
from collections import defaultdict
from utils.build_graphs import build_graphs
from utils.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d
import cv2
from utils.augmentation import augment_image, augment_image_pair, augment_two_images
from itertools import combinations
from src.model.ngm import UNIV_SIZE


def _pad_perm_mats_with_dustbin(mats, ns_pairs):
    max_n1 = max(n1 for n1, _ in ns_pairs)
    max_n2 = max(n2 for _, n2 in ns_pairs)
    t0 = torch.as_tensor(mats[0])
    batch = len(mats)
    out = t0.new_zeros((batch, max_n1 + 1, max_n2 + 1))
    mask = torch.zeros((batch, max_n1 + 1, max_n2 + 1), dtype=torch.bool)
    for i, (mat, (n1, n2)) in enumerate(zip(mats, ns_pairs)):
        t = torch.as_tensor(mat)
        out[i, :n1 + 1, :n2 + 1] = t[:n1 + 1, :n2 + 1]
        mask[i, :n1 + 1, :n2 + 1] = True
    return out, mask


def _standardize(image, annotation):
    """Resize to 320x320 and center crop to 240x320."""
    h, w = image.shape[:2]
    resized = cv2.resize(image, (320, 320))
    scale_x, scale_y = 320 / w, 320 / h
    annos = [[id_, x * scale_x, y * scale_y] for id_, x, y in annotation]
    crop_h, crop_w = 240, 320
    start_x = (320 - crop_w) // 2
    start_y = (320 - crop_h) // 2
    cropped = resized[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_annos = [
        [id_, x - start_x, y - start_y]
        for id_, x, y in annos
        if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h
    ]
    return cropped, cropped_annos



RESCALE=(320, 240)
SRC_GRAPH_CONSTRUCT="tri"
TGT_GRAPH_CONSTRUCT="tri"
SYM_ADJACENCY=True
NORM_MEANS= [0.485, 0.456, 0.406] 
NORM_STD=[0.229, 0.224, 0.225]
MAX_PROB_SIZE=-1

TYPE = '2GM'
FP16 = False
RANDOM_SEED=145
DATALOADER_NUM=0

# class GMDataset(Dataset):
#     def __init__(self, name, bm, length, using_all_graphs=False, cls=None, problem='2GM', augment=None):
#         # Name of Dataset
#         self.name = name
#         # Benchmark Object
#         self.bm = bm
#         # Whether to use all graphs
#         self.using_all_graphs = using_all_graphs
#         # Object size after resizing
#         self.obj_size = self.bm.obj_resize
#         # Determine if in test mode
#         self.test = True if self.bm.sets == 'test' else False
#         # Determine if augmentation is to be applied. Always augment during training
#         if augment is None:
#             self.augment = self.bm.sets == 'train'
        
#         # Class selection
#         self.classes = self.bm.classes if cls in ['none', 'all'] else [cls]

#         self.problem_type = problem

#         if len(self.classes) > 0:
#             self.img_num_list = self.bm.compute_img_num(self.classes[0])

#         # All are classification tasks
#         pairs, total_len = self.bm.get_rand_id_combination()

#         if self.bm.sets == 'test':
#             # In test mode always use the full set of pairs
#             self.length = total_len
#         else:
#             if length is not None and length < total_len:
#                 pairs[0] = pairs[0][:length]
#                 self.length = length
#             else:
#                 self.length = total_len

#         self.id_combination = pairs
#         self.length_list = [self.length]
        

class GMDataset(Dataset):
    def __init__(self, name, bm, length, using_all_graphs=False, cls=None, problem='2GM', augment=None):
        self.name = name
        self.bm = bm
        self.using_all_graphs = using_all_graphs
        self.obj_size = self.bm.obj_resize
        self.test = True if self.bm.sets == 'test' else False
        if augment is None:
            self.augment = self.bm.sets == 'train'
        else:
            self.augment = augment
        self.cls = None if cls in ['none', 'all'] else cls

        if self.cls is None:
            self.classes = self.bm.classes # This is 148
        else:
            self.classes = [self.cls]

        self.problem_type = problem
        # print(f"Classes: {self.classes}")
        # print(f"Benchmark type: {type(self.bm)}")
        if len(self.classes) > 0:
            self.img_num_list = self.bm.compute_img_num(self.classes[0])
        else:
            print("Error: self.classes is empty!")
        # For classification we rely on the genuine/imposter pairs
        pairs, total_len = self.bm.get_rand_id_combination()

        if self.bm.sets == 'test':
            # In test mode always use the full set of pairs
            self.length = total_len
        else:
            # ``length`` may request a subset of pairs during training
            if length is not None and length < total_len:
                pairs[0] = pairs[0][:length]
                self.length = length
            else:
                self.length = total_len

        self.id_combination = pairs
        self.length_list = [self.length]

        # Prebuild image transform once to avoid per-sample Compose creation
        self._img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEANS, NORM_STD)
        ])

    @staticmethod
    def _canonicalize_label(label):
        canonical = str(label)
        # Keep keypoint identity intact; only remove split/session source.
        # If label encodes image augmentation before a keypoint index
        # (e.g. "..._aug_2_57"), normalize it to "..._57".
        canonical = re.sub(r"_aug_\d+_(\d+)$", r"_\1", canonical)
        canonical = re.sub(r"^[RrSs]\d+_", "", canonical)
        return canonical

    @classmethod
    def _build_perm_mat_from_annos(cls, annos1, annos2):
        n1, n2 = len(annos1), len(annos2)
        perm_mat = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)

        label_to_rows = defaultdict(list)
        label_to_cols = defaultdict(list)
        for i, (lab, _, _) in enumerate(annos1):
            canonical = cls._canonicalize_label(lab)
            if canonical == "outlier":
                continue
            label_to_rows[canonical].append(i)
        for j, (lab, _, _) in enumerate(annos2):
            canonical = cls._canonicalize_label(lab)
            if canonical == "outlier":
                continue
            label_to_cols[canonical].append(j)

        matched_rows = set()
        matched_cols = set()
        for label, rows in label_to_rows.items():
            cols = label_to_cols.get(label)
            if not cols:
                continue
            pair_count = min(len(rows), len(cols))
            for k in range(pair_count):
                i = rows[k]
                j = cols[k]
                perm_mat[i, j] = 1.0
                matched_rows.add(i)
                matched_cols.add(j)

        if n1 > 0:
            unmatched_rows = [i for i in range(n1) if i not in matched_rows]
            if unmatched_rows:
                perm_mat[unmatched_rows, n2] = 1.0
        if n2 > 0:
            unmatched_cols = [j for j in range(n2) if j not in matched_cols]
            if unmatched_cols:
                perm_mat[n1, unmatched_cols] = 1.0

        return perm_mat, len(matched_rows)

    def _augment_or_standardize_pair_same(self, image, annos, clip_to_univ: bool = False):
        """Return two views from the same image and filtered annotations."""
        if self.augment:
            (img1, ann1), (img2, ann2) = augment_image_pair(
                image, annos, min_points=5, min_common=4, max_attempts=5, n_jobs=2
            )
        else:
            img1, ann1 = _standardize(image, annos)
            img2, ann2 = _standardize(image, annos)

        if clip_to_univ:
            ann1 = ann1[:UNIV_SIZE]
            ann2 = ann2[:UNIV_SIZE]
        return (img1, ann1), (img2, ann2)

    def _augment_or_standardize_pair_diff(self, img1_orig, ann1_base, img2_orig, ann2_base, clip_to_univ: bool = False):
        """Return two views from two different images and annotations."""
        if self.augment:
            (img1, ann1), (img2, ann2) = augment_two_images(
                img1_orig, ann1_base, img2_orig, ann2_base, min_points=5, n_jobs=2
            )
        else:
            img1, ann1 = _standardize(img1_orig, ann1_base)
            img2, ann2 = _standardize(img2_orig, ann2_base)

        if clip_to_univ:
            if len(ann1) > UNIV_SIZE:
                ann1 = ann1[:UNIV_SIZE]
            if len(ann2) > UNIV_SIZE:
                ann2 = ann2[:UNIV_SIZE]
        return (img1, ann1), (img2, ann2)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == '2GM':
            return self.get_pair(idx)
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

    def get_pair(self, idx):
        """Return a pair of graphs for classification (genuine/imposter)."""
        pair = self.id_combination[0][idx % self.length]

        
        result = self.bm.get_data(list(pair))
        if len(result) == 3:
            anno_pair, _, id_list = result
        else:
            anno_pair, id_list = result

        
        cls = [anno['cls'] for anno in anno_pair]

        # Determine label: 1 if genuine, 0 if imposter
        fid0 = self.bm._finger_id(cls[0]) if hasattr(self.bm, '_finger_id') else cls[0]
        fid1 = self.bm._finger_id(cls[1]) if hasattr(self.bm, '_finger_id') else cls[1]
        label = 1 if fid0 == fid1 else 0

        img_path1 = self.bm.get_path(pair[0])
        img_path2 = self.bm.get_path(pair[1])
        img1_orig = cv2.imread(img_path1)
        img2_orig = cv2.imread(img_path2)
        annos1_base = [[kp['labels'], kp['x'], kp['y']] for kp in anno_pair[0]['kpts']]
        annos2_base = [[kp['labels'], kp['x'], kp['y']] for kp in anno_pair[1]['kpts']]
        (img1, annos1_filtered), (img2, annos2_filtered) = self._augment_or_standardize_pair_diff(
            img1_orig,
            annos1_base,
            img2_orig,
            annos2_base,
            clip_to_univ=True,
        )
        perm_mat, n_common = self._build_perm_mat_from_annos(annos1_filtered, annos2_filtered)

        # Genuine pairs should carry valid positive correspondences. If random
        # augmentation removes all overlap, fall back to a same-image dual view.
        if label == 1 and n_common == 0:
            (img1, annos1_filtered), (img2, annos2_filtered) = self._augment_or_standardize_pair_same(
                img1_orig,
                annos1_base,
                clip_to_univ=True,
            )
            perm_mat, n_common = self._build_perm_mat_from_annos(annos1_filtered, annos2_filtered)

        

        P1 = np.asarray([[x, y] for _, x, y in annos1_filtered], dtype=np.float32).reshape(-1, 2)
        P2 = np.asarray([[x, y] for _, x, y in annos2_filtered], dtype=np.float32).reshape(-1, 2)
        
        n1, n2 = len(P1), len(P2)

        # if not label:
        #     print(annos1_filtered, annos2_filtered)
        #     print("label", label, "n_common:", n_common)
        #     print("P1 shape:", P1.shape, "P2 shape:", P2.shape)

        A1, G1, H1, e1 = build_graphs(P1, n1, stg=SRC_GRAPH_CONSTRUCT, sym=SYM_ADJACENCY)
        if TGT_GRAPH_CONSTRUCT == 'same':
            if perm_mat.sum() == 0:
                A2, G2, H2, e2 = build_graphs(P2, n2, stg=SRC_GRAPH_CONSTRUCT, sym=SYM_ADJACENCY)
            else:
                G2 = perm_mat.transpose().dot(G1)
                H2 = perm_mat.transpose().dot(H1)
                A2 = G2.dot(H2.transpose())
                e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(P2, n2, stg=TGT_GRAPH_CONSTRUCT, sym=SYM_ADJACENCY)

        pyg_graph1 = self.to_pyg_graph(A1, P1)
        pyg_graph2 = self.to_pyg_graph(A2, P2)

        imgs = [img1, img2]
        if imgs[0] is not None:
            imgs = [self._img_transform(img) for img in imgs]

        ret_dict = {
            'Ps': [torch.Tensor(x) for x in [P1, P2]],
            'ns': [torch.tensor(x) for x in [n1, n2]],
            'es': [torch.tensor(x) for x in [e1, e2]],
            'gt_perm_mat': perm_mat,
            'Gs': [torch.Tensor(x) for x in [G1, G2]],
            'Hs': [torch.Tensor(x) for x in [H1, H2]],
            'As': [torch.Tensor(x) for x in [A1, A2]],
            'pyg_graphs': [pyg_graph1, pyg_graph2],
            'cls': [str(x) for x in cls],
            'id_list': id_list,
            'univ_size': torch.tensor(n_common),
            'images': imgs,
            'label': torch.tensor(label, dtype=torch.float32)
        }

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
            ns_pairs = None
            if 'gt_perm_mat' in inp[0] and 'ns' in inp[0]:
                ns_pairs = []
                for d in inp:
                    n1, n2 = d['ns']
                    n1 = int(n1.item()) if isinstance(n1, torch.Tensor) else int(n1)
                    n2 = int(n2.item()) if isinstance(n2, torch.Tensor) else int(n2)
                    ns_pairs.append((n1, n2))
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                k = ks[0]
                if k == 'gt_perm_mat' and ns_pairs is not None:
                    has_dustbin = True
                    for mat, (n1, n2) in zip(vs, ns_pairs):
                        shape = mat.shape
                        if shape[0] != n1 + 1 or shape[1] != n2 + 1:
                            has_dustbin = False
                            break
                    if has_dustbin:
                        padded, mask = _pad_perm_mats_with_dustbin(vs, ns_pairs)
                        ret[k] = padded
                        ret['gt_perm_mat_mask'] = mask
                        continue
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif isinstance(inp[0], pyg.data.Data):
            ret = pyg.data.Batch.from_data_list(inp)
        else:
            ret = inp
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
    # For univ_size, if it's a list of scalar tensors, stack them.
    if isinstance(ret['univ_size'], list):
        ret['univ_size'] = torch.stack(ret['univ_size'])
    for v in ret.values():
        if isinstance(v, list):
            ret['num_graphs'] = len(v)
            break
    ret['batch_size'] = len(data)
    # For univ_size, stack scalar tensors
    if isinstance(ret['univ_size'], list):
        ret['univ_size'] = torch.stack(ret['univ_size'])
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


def get_dataloader(dataset, batch_size, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
