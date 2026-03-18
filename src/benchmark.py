import os
import tempfile
import json
from pathlib import Path
from pygmtools.benchmark import Benchmark  # Import the original Benchmark class
# Import your new dataset class.
from src.dataset import L3SFV2AugmentedDataset, PolyUDBII, PolyUDBI, L3SF
from PIL import Image
import random
import itertools
import numpy as np
import re
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod

from src.gmdataset import RESCALE


PAIRING_TASK = "classify"



def _normalize_filter(filter_value):
    if filter_value is None:
        return None
    if isinstance(filter_value, str):
        normalized = filter_value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized in {"intersection", "inclusion"}:
            return normalized
    raise ValueError("filter must be one of: None, 'intersection', 'inclusion'")

class ClassifyPairs:
    """Default classification pairing logic based on class grouping."""

    def _finger_id(self, cls_name: str) -> str:
        """Return a canonical finger id across split/session variants."""
        fid = str(cls_name)
        # Remove dataset split/session prefixes (e.g. R1_, R5_, S2_).
        fid = re.sub(r"^[RrSs]\d+_", "", fid)
        # Remove augmentation suffixes if present.
        fid = re.sub(r"_aug_\d+$", "", fid)
        return fid

    def _build_classify_pairs(self, only_genuine=False):
        """Generate genuine and imposter pairs for the classification task."""
        # Prefer the in-memory annotations already loaded by FingerprintBenchmarkBase
        data_dict = getattr(self, "data_dict", None)
        if not data_dict:
            json_path = os.path.join(
                self.dataset_dir, f"{self.sets}-{self.suffix}.json"
            )
            with open(json_path, 'r') as f:
                data_dict = json.load(f)

        from collections import defaultdict

        groups = defaultdict(list)
        for img_id, anno in data_dict.items():
            cls_name = anno.get('cls')
            if not cls_name:
                continue
            fid = self._finger_id(cls_name)
            groups[fid].append(img_id)

        # Genuine matches: prefer cross-image pairs from the same finger.
        # Fall back to self-pair only when there is a single sample.
        genuine_pairs = []
        for id_list in groups.values():
            if not id_list:
                continue
            # Keep deterministic ordering while removing accidental duplicates.
            uniq_ids = list(dict.fromkeys(id_list))
            if len(uniq_ids) >= 2:
                genuine_pairs.extend(list(itertools.combinations(uniq_ids, 2)))
            else:
                genuine_pairs.append((uniq_ids[0], uniq_ids[0]))

        if only_genuine:
            random.shuffle(genuine_pairs)
            return genuine_pairs

        # Imposter matches: take one representative per finger and pair them uniquely
        representatives = [id_list[0] for id_list in groups.values() if id_list]
        imposter_pairs = list(itertools.combinations(representatives, 2))

        if self.sets == 'test':
            # In test mode return all pairs without balancing the counts
            return genuine_pairs + imposter_pairs

        # Balance the number of pairs so we have equal genuine and imposter
        pair_count = min(len(genuine_pairs), len(imposter_pairs))
        if pair_count == 0:
            return []
        pairs = random.sample(
            random.sample(genuine_pairs, pair_count) +
            random.sample(imposter_pairs, pair_count),
            2 * pair_count
        )

        #Use all pairs   
        # pairs = random.sample(
        #     genuine_pairs + imposter_pairs,
        #     len(genuine_pairs) + len(imposter_pairs)
        # )

        return pairs


class SessionStancePairMixin(ABC):
    """Pairing logic for datasets with person/session/stance identifiers."""

    @abstractmethod
    def _parse_id(self, img_id):
        raise NotImplementedError

    def _build_classify_pairs(self):
        """Generate genuine and imposter pairs according to session/stance protocol."""
        # Build compact structures for speed.
        person_lists = {}
        s1_lists = {}
        s2_lists = {}
        s1_stance1 = {}
        s2_stance1 = {}
        for img_id in self.data_dict.keys():
            parsed_info = self._parse_id(img_id)
            if not parsed_info:
                continue
            person, session, stance = parsed_info
            person_lists.setdefault(person, []).append(img_id)
            # If session 1, add to s1 list else s2 list
            if session == 1:
                s1_lists.setdefault(person, []).append(img_id)
                if stance == 1 and person not in s1_stance1:
                    s1_stance1[person] = img_id
            elif session == 2:
                s2_lists.setdefault(person, []).append(img_id)
                if stance == 1 and person not in s2_stance1:
                    s2_stance1[person] = img_id

        genuine_pairs = []
        if self.sets == 'train':
            for person_ids in person_lists.values():
                uniq_ids = list(dict.fromkeys(person_ids))
                if len(uniq_ids) >= 2:
                    genuine_pairs.extend(itertools.combinations(uniq_ids, 2))
                elif uniq_ids:
                    genuine_pairs.append((uniq_ids[0], uniq_ids[0]))
        else:
            # Combine Genuine pairs from s1 and s2 lists
            for person, s1 in s1_lists.items():
                s2 = s2_lists.get(person)
                if not s2:
                    continue
                for id1 in s1:
                    for id2 in s2:
                        genuine_pairs.append((id1, id2))
        if self.only_genuine   :
            random.shuffle(genuine_pairs)
            return genuine_pairs

        # For the first stance only, create imposter pairs
        imposter_pairs = []
        persons = list(set(s1_stance1) | set(s2_stance1))
        for i, pa in enumerate(persons):
            id_a = s1_stance1.get(pa)
            if id_a is None:
                continue
            for pb in persons[i + 1:]:
                id_b = s2_stance1.get(pb)
                if id_b is not None:
                    imposter_pairs.append((id_a, id_b))
                id_a2 = s1_stance1.get(pb)
                id_b2 = s2_stance1.get(pa)
                if id_a2 is not None and id_b2 is not None:
                    imposter_pairs.append((id_a2, id_b2))

        pairs = genuine_pairs + imposter_pairs
        if not pairs:
            return []

        if self.sets == 'test':
            random.shuffle(pairs)
            return pairs

        pair_count = min(len(genuine_pairs), len(imposter_pairs))
        if pair_count == 0:
            return []
        random.shuffle(genuine_pairs)
        random.shuffle(imposter_pairs)
        pairs = genuine_pairs[:pair_count] + imposter_pairs[:pair_count]
        random.shuffle(pairs)
        return pairs


class PolyUIdParseMixin:
    """ID parser for PolyU DBI/DBII identifiers."""

    def _parse_id(self, img_id):
        """Parse an image identifier into (person, session, stance)."""
        parts = img_id.split('_')
        if len(parts) < 3:
            return None
        try:
            person = int(parts[0])
            session = int(parts[1])
            stance = int(parts[2])
            return person, session, stance
        except ValueError:
            return None


class FingerprintBenchmarkBase(Benchmark, ABC):
    """Shared benchmark logic for fingerprint datasets."""

    @abstractmethod
    def _build_classify_pairs(self):
        raise NotImplementedError

    def __init__(self, sets, obj_resize=RESCALE, problem='2GM',
                 filter=None, task='classify', dataset_cls=L3SFV2AugmentedDataset,
                 name=None, only_genuine=False, **args):
        task = PAIRING_TASK
        self.only_genuine = only_genuine
        # Instead of a dataset name from a fixed list, we use our new dataset.
        self.name = name if name is not None else getattr(dataset_cls, '__name__', 'CustomDataset')
        self.problem = problem
        self.filter = _normalize_filter(filter)
        self.obj_resize = obj_resize

        # Instantiate the dataset using the unified pairing protocol.
        # Ensure the JSON annotations exist by invoking ``to_json``.
        dataset_instance = dataset_cls(
            sets,
            obj_resize,
            task=task,
            **args,
        )
        # Ensure on-disk annotations are prepared
        try:
            json_path = dataset_instance.to_json()
        except Exception:
            # Fallback: proceed and let the later open() raise with context
            json_path = None

        self.task = dataset_instance.task

        self.sets = sets

        # If your dataset class does not already define these, set them here:
        if not hasattr(dataset_instance, "dataset_dir"):
            # Prefer dataset-specific output_dir if present
            out_dir = getattr(dataset_instance, "output_dir", Path(f"data/{self.name}"))
            dataset_instance.dataset_dir = str(out_dir)
        if not hasattr(dataset_instance, "suffix"):
            dataset_instance.suffix = f"{obj_resize}"

        # Prefer the path returned by to_json; otherwise construct it
        json_file = str(json_path) if json_path is not None else os.path.join(
            dataset_instance.dataset_dir,
            f"{sets}-{dataset_instance.suffix}.json",
        )
        if not hasattr(dataset_instance, "classes"):
            # Derive the list of classes by reading the JSON file if the dataset
            # did not already provide them.
            with open(json_file, "r") as f:
                data_dict = json.load(f)

            dataset_instance.classes = list({data_dict[k]['cls'] for k in data_dict})

        self.classes = dataset_instance.classes
        self.dataset_dir = dataset_instance.dataset_dir
        self.suffix = dataset_instance.suffix

        # Build the paths for the unified data interface.
        self.data_path = json_file
        self.data_list_path = json_file

        # Load the data dictionary from the JSON file.
        with open(self.data_path, "r") as f:
            self.data_dict = json.load(f)

        # ``data_dict`` now contains only the annotations from the selected
        # split.  When generating classification pairs we will ensure the IDs
        # come from this dictionary so there is no need to merge data from other
        # splits.

        if self.sets == 'test':
            tmpfile = tempfile.gettempdir()
            pid_num = os.getpid()
            cache_dir = str(pid_num) + '_gt_cache'
            self.gt_cache_path = os.path.join(tmpfile, cache_dir)

            if not os.path.exists(self.gt_cache_path):
                os.mkdir(self.gt_cache_path)
                print('gt perm mat cache built')


    def get_path(self, id):
        return self.data_dict[id]["path"]

    def get_data(self, ids, test=False, shuffle=True):
        r"""
        Fetch a data pair or pairs of data by image ID for training or test.

        :param ids: list of image ID, usually in ``train.json`` or ``test.json``
        :param test: bool, whether the fetched data is used for test; if true, this function will not return ground truth
        :param shuffle: bool, whether to shuffle the order of keypoints
        :return:
                    **data_list**: list of data, like ``[{'img': np.array, 'kpts': coordinates of kpts}, ...]``

                    **perm_mat_dict**: ground truth, like ``{(0,1):scipy.sparse, (0,2):scipy.sparse, ...}``, ``(0,1)`` refers to data pair ``(ids[0],ids[1])``

                    **ids**: list of image ID
        """

        ids.sort()
        # print( "Number of ids:", len(ids))
        data_list = []
        for keys in ids:
            obj_dict = dict()
            # boundbox = self.data_dict[keys]['bounds']
            img_file = self.data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                img = img.convert("RGB")
                img.load()          # FORCE into memory
                obj = img.copy()    # detach from file

                if self.name == 'CUB2011':
                    if not obj.mode == 'RGB':
                        obj = obj.convert('RGB')
            try:
                obj_dict['img'] = np.array(obj)
            except Exception as e:
                print("Image load failed:", ids)
                print("Image object:", obj)
                print("Error:", e)
                raise
            obj_dict['kpts'] = self.data_dict[keys]['kpts']
            obj_dict['cls'] = self.data_dict[keys]['cls']
            obj_dict['univ_size'] = self.data_dict[keys]['univ_size']
            if shuffle:
                random.shuffle(obj_dict['kpts'])
            data_list.append(obj_dict)

        
        # print("The class is ", obj_dict['cls'])
        perm_mat_dict = dict()
        id_combination = list(itertools.combinations(list(range(len(ids))), 2))
        for id_tuple in id_combination:
            # No filter for different classes
            if data_list[id_tuple[0]]['cls'] != data_list[id_tuple[1]]['cls']:
                # perm_mat = np.zeros(
                #     (len(data_list[id_tuple[0]]['kpts'])+1, len(data_list[id_tuple[1]]['kpts'])+1),
                #     dtype=np.float32,
                # )
                n_kpts_a = len(data_list[id_tuple[0]]['kpts'])
                n_kpts_b = len(data_list[id_tuple[1]]['kpts'])
                max_kpts = n_kpts_a if n_kpts_a >= n_kpts_b else n_kpts_b
                size = max_kpts + 1  # dustbin
                perm_mat = np.zeros((size, size), dtype=np.float32)

                perm_mat[-1, :] = 1.0
                perm_mat[:, -1] = 1.0
                perm_mat[-1, -1] = 0.0

                if not (len(ids) > 2 and self.filter == 'intersection'):
                    perm_mat_dict[id_tuple] = coo_matrix(perm_mat)
                continue

            # Vectorized label comparison
            a_labels = np.array([kp['labels'] for kp in data_list[id_tuple[0]]['kpts']])
            b_labels = np.array([kp['labels'] for kp in data_list[id_tuple[1]]['kpts']])

            eq = a_labels[:, None] == b_labels[None, :]
            perm_mat = (eq & (a_labels[:, None] != 'outlier')).astype(np.float32)

            row_list = np.nonzero(eq.any(axis=1))[0].tolist()
            col_list = np.nonzero(eq.any(axis=0))[0].tolist()

            row_list.sort()
            col_list.sort()
            if self.filter == 'intersection':
                perm_mat = perm_mat[row_list, :]
                perm_mat = perm_mat[:, col_list]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
                data_list[id_tuple[1]]['kpts'] = [data_list[id_tuple[1]]['kpts'][i] for i in col_list]
            elif self.filter == 'inclusion':
                perm_mat = perm_mat[row_list, :]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
            
            if not (len(ids) > 2 and self.filter == 'intersection'):
                sparse_perm_mat = coo_matrix(perm_mat)
                perm_mat_dict[id_tuple] = sparse_perm_mat

        if len(ids) > 2 and self.filter == 'intersection':
            for p in range(len(ids) - 1):
                perm_mat_list = [np.zeros([len(data_list[p]['kpts']), len(x['kpts'])], dtype=np.float32) for x in
                                 data_list[p + 1: len(ids)]]
                row_list = []
                col_lists = []
                for i in range(len(ids) - p - 1):
                    col_lists.append([])

                for i, keypoint in enumerate(data_list[p]['kpts']):
                    kpt_idx = []
                    for anno_dict in data_list[p + 1: len(ids)]:
                        kpt_name_list = [x['labels'] for x in anno_dict['kpts']]
                        if keypoint['labels'] in kpt_name_list:
                            kpt_idx.append(kpt_name_list.index(keypoint['labels']))
                        else:
                            kpt_idx.append(-1)
                    row_list.append(i)
                    for k in range(len(ids) - p - 1):
                        j = kpt_idx[k]
                        if j != -1:
                            col_lists[k].append(j)
                            if keypoint['labels'] != 'outlier':
                                perm_mat_list[k][i, j] = 1

                row_list.sort()
                for col_list in col_lists:
                    col_list.sort()

                for k in range(len(ids) - p - 1):
                    perm_mat_list[k] = perm_mat_list[k][row_list, :]
                    perm_mat_list[k] = perm_mat_list[k][:, col_lists[k]]
                    id_tuple = (p, k + p + 1)
                    perm_mat_dict[id_tuple] = coo_matrix(perm_mat_list[k])

        if self.sets == 'test':
            for pair in id_combination:
                id_pair = (ids[pair[0]], ids[pair[1]])
                gt = perm_mat_dict[pair].toarray()
                gt_path = os.path.join(self.gt_cache_path, str(id_pair) + '_' + str(gt.shape[0]) + '_'
                                       + str(gt.shape[1]) + '.npy')
                if not os.path.exists(gt_path):
                    np.save(gt_path, perm_mat_dict[pair])

        if not test:
            return data_list, perm_mat_dict, ids
        else:
            return data_list, ids


    def compute_length(self, cls=None, num=2):
        r"""
        Compute the length of image combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for two-graph matching problem
        :return: length of combinations
        """
        # TODO: modify this compute length to give correct length for classes. However it should also be the length during classification we will no longer be doing a matching task.
        if self.task == 'classify':
            if not hasattr(self, '_classify_pairs'):
                self._classify_pairs = self._build_classify_pairs(only_genuine=self.only_genuine)
            return len(self._classify_pairs)

        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0

        if clss is not None:
            if self.name != 'SPair71k':
                data_list = []
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                # Adjust combination size:
                # If there are fewer than "num" samples, use the actual count.
                actual_num = num if len(data_list) >= num else len(data_list)
                id_combination = list(itertools.combinations(data_list, actual_num))
                length += len(id_combination)
            else:
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        length += 1
        else:
            for clss in self.classes:
                if self.name != 'SPair71k':
                    data_list = []
                    for id in data_id:
                        if self.data_dict[id]['cls'] == clss:
                            data_list.append(id)
                    actual_num = num if len(data_list) >= num else len(data_list)
                    id_combination = list(itertools.combinations(data_list, actual_num))
                    length += len(id_combination)
                else:
                    for id_pair in data_id:
                        if self.data_dict[id_pair[0]]['cls'] == clss:
                            length += 1
        return length


    def get_rand_id_combination(self, num=2, combinations_per_class=1):
        """
        Generate combinations by selecting one ID from different classes

        Args:

            num: Number of IDs to combine (default 2 for pairs)
            combinations_per_class: Number of combinations to generate per class

        Returns:
            id_combination_list: Nested list of ID combinations
            length: Total number of combinations
        """
    
        if not hasattr(self, '_classify_pairs'):
            self._classify_pairs = self._build_classify_pairs()
        return [self._classify_pairs], len(self._classify_pairs)


class L3SFV2AugmentedBenchmark(ClassifyPairs, FingerprintBenchmarkBase):
    """Benchmark wrapper for the :class:`L3SFV2AugmentedDataset` using unified pairing."""

    pass

class L3SFBenchmark(SessionStancePairMixin, FingerprintBenchmarkBase):
    """Benchmark for the L3SF dataset with classification pair logic."""

    def __init__(self, sets, obj_resize=RESCALE, problem='2GM',
                 filter=None, task='match', **args):
        super().__init__(
            sets,
            obj_resize=obj_resize,
            problem=problem,
            filter=filter,
            task=task,
            dataset_cls=L3SF,
            name="L3-SF",
            **args,
        )

    def _parse_id(self, img_id):
        """Parse ``R<fold>_<person>_<session>_<stance>`` identifiers."""
        parts = img_id.split('_')
        if len(parts) != 4:
            return None
        try:
            person = f"{parts[0]}_{parts[1]}"
            session = int(parts[2])
            stance = int(parts[3])
            return person, session, stance
        except ValueError:
            return None


class PolyUDBIIBenchmark(PolyUIdParseMixin, SessionStancePairMixin, FingerprintBenchmarkBase):
    """Benchmark for the PolyU DBII dataset with classification pair logic."""

    def __init__(self, sets, obj_resize=RESCALE, problem='2GM',
                 filter=None, task='match', **args):
        super().__init__(
            sets,
            obj_resize=obj_resize,
            problem=problem,
            filter=filter,
            task=task,
            dataset_cls=PolyUDBII,
            name="PolyU-DBII",
            **args,
        )


class PolyUDBIBenchmark(PolyUIdParseMixin, SessionStancePairMixin, FingerprintBenchmarkBase):
    """Benchmark for the PolyU DBI dataset with classification pair logic."""

    def __init__(self, sets, obj_resize=RESCALE, problem='2GM',
                 filter=None, task='match', **args):
        super().__init__(
            sets,
            obj_resize=obj_resize,
            problem=problem,
            filter=filter,
            task=task,
            dataset_cls=PolyUDBI,
            name="PolyU-DBI",
            **args,
        )

    pass


def _select_ids(benchmark, pair_size=2):
    if benchmark.task == 'classify':
        combinations, _ = benchmark.get_rand_id_combination(num=pair_size)
        if not combinations:
            return []
        first_group = combinations[0]
        if not first_group:
            return []
        return list(first_group[0])
    ids = list(benchmark.data_dict.keys())
    return ids[:pair_size]


def _print_data_model(data_list, perm_mat_dict, ids):
    print(f"ids: {ids}")
    for idx, item in enumerate(data_list):
        img = item.get("img")
        img_shape = getattr(img, "shape", None)
        kpts = item.get("kpts", [])
        cls_name = item.get("cls")
        print(f"item[{idx}] cls={cls_name} img_shape={img_shape} kpts={len(kpts)} univ={item.get('univ_size')}")
    for pair, mat in perm_mat_dict.items():
        dense = mat.toarray() if hasattr(mat, "toarray") else mat
        shape = getattr(dense, "shape", None)
        nonzero = int(np.count_nonzero(dense)) if dense is not None else 0
        print(f"perm[{pair}] shape={shape} nnz={nonzero}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick benchmark get_data sanity check.")
    parser.add_argument(
        "--dataset",
        choices=("l3sfv2", "l3sf", "polyu-dbi", "polyu-dbii"),
        default="l3sfv2",
        help="Benchmark dataset to test.",
    )
    parser.add_argument("--sets", choices=("train", "val", "test"), default="train")
    parser.add_argument("--task", choices=("match", "classify"), default="match")
    parser.add_argument("--train-root", default=None, help="Root dir for dataset (train/val/test subdirs if used).")
    parser.add_argument("--obj-resize", type=int, nargs=2, default=RESCALE, metavar=("W", "H"))
    parser.add_argument("--pair-size", type=int, default=2)
    args = parser.parse_args()

    bm_cls = {
        "l3sfv2": L3SFV2AugmentedBenchmark,
        "l3sf": L3SFBenchmark,
        "polyu-dbi": PolyUDBIBenchmark,
        "polyu-dbii": PolyUDBIIBenchmark,
    }[args.dataset]


    bm_kwargs = {}
    if args.train_root is not None:
        bm_kwargs["train_root"] = args.train_root

    benchmark = bm_cls(
        sets=args.sets,
        obj_resize=tuple(args.obj_resize),
        task=args.task,
        **bm_kwargs,
    )

    ids = _select_ids(benchmark, pair_size=args.pair_size)
    if len(ids) < args.pair_size:
        print(f"Not enough ids to build a pair. found={len(ids)} needed={args.pair_size}")
        return

    data_list, perm_mat_dict, ids = benchmark.get_data(ids, test=False, shuffle=False)
    _print_data_model(data_list, perm_mat_dict, ids)


if __name__ == "__main__":
    main()
