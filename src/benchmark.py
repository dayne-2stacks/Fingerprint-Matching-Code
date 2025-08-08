import os
import tempfile
import json
from pathlib import Path
from pygmtools.benchmark import Benchmark  # Import the original Benchmark class
# Import your new dataset class.
# (Adjust the import according to where your L3SFV2AugmentedDataset is defined.)
from src.dataset import L3SFV2AugmentedDataset, PolyUDBII, PolyUDBI
from PIL import Image
import random
import itertools
import numpy as np
from scipy.sparse import coo_matrix

class L3SFV2AugmentedBenchmark(Benchmark):
    """Benchmark wrapper for the :class:`L3SFV2AugmentedDataset`.

    The class behaves like the original :class:`pygmtools.benchmark.Benchmark`
    but adds support for two tasks: ``match`` and ``classify``.  The behaviour
    defaults to the standard graph matching benchmark when ``task`` is
    ``'match'``.  When ``task`` is ``'classify'`` it generates image
    combinations following the fingerprint verification protocol described in
    the paper.
    """

    def __init__(self, sets, obj_resize=(512, 512), problem='2GM',
                 filter='intersection', task='match', dataset_cls=L3SFV2AugmentedDataset,
                 name=None, **args):
        # Instead of a dataset name from a fixed list, we use our new dataset.
        self.name = name if name is not None else getattr(dataset_cls, '__name__', 'CustomDataset')
        self.problem = problem
        self.filter = filter
        self.obj_resize = obj_resize

        # Instantiate the dataset. ``task`` controls whether we operate in
        # matching or classification mode.
        # (The dataset ``__init__`` is expected to generate the JSON files if
        # they do not exist.)
        dataset_instance = dataset_cls(
            sets,
            obj_resize,
            task=task,
            **args,
        )
        
        self.task = dataset_instance.task
        # When operating in classification mode, keep the split passed by the
        # caller rather than forcing the benchmark to "test".  This ensures the
        # benchmark only references IDs that belong to the selected split.
        self.sets = sets
            

        # Make sure that the dataset has the attributes that Benchmark expects.
        # For example, Benchmark later uses:
        #   - dataset_dir: a directory where JSON annotation files are saved.
        #   - suffix: a string suffix used in the JSON filenames.
        #   - classes: a list of available object classes.
        #
        # If your dataset class does not already define these, set them here:
        if not hasattr(dataset_instance, "dataset_dir"):
            dataset_instance.dataset_dir = f"data/{self.name}"
        if not hasattr(dataset_instance, "suffix"):
            dataset_instance.suffix = f"{obj_resize}"

        json_file = os.path.join(
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
            
        if self.sets == 'test' or self.task == 'classify':
            tmpfile = tempfile.gettempdir()
            pid_num = os.getpid()
            cache_dir = str(pid_num) + '_gt_cache'
            self.gt_cache_path = os.path.join(tmpfile, cache_dir)

            if not os.path.exists(self.gt_cache_path):
                os.mkdir(self.gt_cache_path)
                print('gt perm mat cache built')

    
    def get_path(self, id):
        return self.data_dict[id]["path"]

    # ------------------------------------------------------------------
    # Helper utilities for classification task
    # ------------------------------------------------------------------
    def _finger_id(self, cls_name: str) -> str:
        """Return the finger identifier without the session prefix."""
        # return cls_name.split('_', 1)[1] if '_' in cls_name else cls_name
        return cls_name

    def _build_classify_pairs(self):
        """Generate genuine and imposter pairs for the classification task."""
        json_path = os.path.join(
            self.dataset_dir, f"{self.sets}-{self.suffix}.json"
        )
        with open(json_path, 'r') as f:
            data_dict = json.load(f)

        from collections import defaultdict

        groups = defaultdict(list)
        for k, v in data_dict.items():
            fid = self._finger_id(v['cls'])
            groups[fid].append(k)

        genuine_pairs = []
        imposter_pairs = []

        # Genuine matches: pair each image with itself so two augmented copies
        for id_list in groups.values():
            for img_id in id_list:
                genuine_pairs.append((img_id, img_id))

        # Imposter matches: one representative from each finger paired with the
        # representative of every other finger
        fids = list(groups.keys())
        for i, fid in enumerate(fids):
            if not groups[fid]:
                continue
            base = groups[fid][0]
            for j, other_fid in enumerate(fids):
                if other_fid == fid or not groups[other_fid]:
                    continue
                imposter_pairs.append((base, groups[other_fid][0]))

        if self.sets == 'test':
            # In test mode return all pairs without balancing the counts
            pairs = genuine_pairs + imposter_pairs
        else:
            # Balance the number of pairs so we have equal genuine and imposter
            pair_count = min(len(genuine_pairs), len(imposter_pairs))
            pairs = genuine_pairs[:pair_count] + imposter_pairs[:pair_count]

        return pairs
    
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
        data_list = []
        for keys in ids:
            obj_dict = dict()
            # boundbox = self.data_dict[keys]['bounds']
            img_file = self.data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                #obj = img.resize(self.obj_resize, resample=Image.BICUBIC,
                #                 box=(boundbox[0], boundbox[1], boundbox[2], boundbox[3]))
                obj=img
                if self.name == 'CUB2011':
                    if not obj.mode == 'RGB':
                        obj = obj.convert('RGB')
            obj_dict['img'] = np.array(obj)
            obj_dict['kpts'] = self.data_dict[keys]['kpts']
            obj_dict['cls'] = self.data_dict[keys]['cls']
            obj_dict['univ_size'] = self.data_dict[keys]['univ_size']
            if shuffle:
                random.shuffle(obj_dict['kpts'])
            data_list.append(obj_dict)

        if self.task == 'classify':
            # if not test:
            #     return data_list, {}, ids
            return data_list, ids

        perm_mat_dict = dict()
        id_combination = list(itertools.combinations(list(range(len(ids))), 2))
        for id_tuple in id_combination:
            perm_mat = np.zeros([len(data_list[_]['kpts']) for _ in id_tuple], dtype=np.float32)
            row_list = []
            col_list = []

            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        if keypoint['labels'] != 'outlier':
                            perm_mat[i, j] = 1
            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        row_list.append(i)
                        break
            for i, keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        col_list.append(i)
                        break
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

        if self.sets == 'test' or self.task == 'classify':
            for pair in id_combination:
                id_pair = (ids[pair[0]], ids[pair[1]])
                gt = perm_mat_dict[pair].toarray()
                gt_path = os.path.join(self.gt_cache_path, str(id_pair) + '_' + str(gt.shape[0]) + '_'
                                       + str(gt.shape[1]) + '.npy')
                if not os.path.exists(gt_path):
                    np.save(gt_path, perm_mat_dict[pair])

        if not test or self.task == 'classify':
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
        if self.task == 'classify':
            if not hasattr(self, '_classify_pairs'):
                self._classify_pairs = self._build_classify_pairs()
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
        if self.task == 'classify':
            if not hasattr(self, '_classify_pairs'):
                self._classify_pairs = self._build_classify_pairs()
            return [self._classify_pairs], len(self._classify_pairs)

        from collections import defaultdict
        length = 0
        id_combination_list = []
        
        with open(self.data_list_path) as f1:
            data_id = json.load(f1)
        
        if len(data_id) < 2:
            # Not enough data to sample
            return [], 0

        available_ids = [id for id in data_id if id in self.data_dict]
        
        # Group IDs by class
        class_to_ids = defaultdict(list)
        for id in available_ids:
            cls = self.data_dict[id].get('cls', '')
            class_to_ids[cls].append(id)
        
        classes = list(class_to_ids.keys())
        
        # Create cross-class combinations
        for base_cls in classes:
            # For each class, create combinations with other classes
            base_ids = class_to_ids[base_cls]
            other_classes = [c for c in classes if c != base_cls]
            
            if len(other_classes) < num-1:
                # Not enough other classes for combination
                continue
            
            # Create combinations_per_class random combinations
            class_combinations = []
            for _ in range(combinations_per_class):
                # Select random base ID
                if not base_ids:
                    continue
                base_id = random.choice(base_ids)
                
                # Select random other classes and IDs
                selected_classes = random.sample(other_classes, num-1)
                combination = [base_id]
                
                # Add one ID from each selected class
                for cls in selected_classes:
                    if class_to_ids[cls]:
                        combination.append(random.choice(class_to_ids[cls]))
                
                # Ensure we have exactly num IDs
                if len(combination) == num:
                    class_combinations.append(tuple(combination))
            
            if class_combinations:
                id_combination_list.append(class_combinations)
                length += len(class_combinations)
        
        return id_combination_list, length
    

class PolyUDBIIBenchmark(L3SFV2AugmentedBenchmark):
    """Benchmark for the PolyU DBII dataset with classification pair logic."""

    def __init__(self, sets, obj_resize=(512, 512), problem='2GM',
                 filter='intersection', task='match', **args):
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

    # ------------------------------------------------------------------
    # PolyU DBII specific pair generation for classification
    # ------------------------------------------------------------------
    def _parse_id(self, img_id):
        """Parse an image identifier into (person, session, stance).

        Expected format: ``DBII_{person}_{session}_{stance}`` with integer
        components. Returns ``(person, session, stance)`` or ``None`` if the
        pattern does not match.
        """
        parts = img_id.split('_')
        if len(parts) < 4:
            return None
        try:
            person = int(parts[1])
            session = int(parts[2])
            stance = int(parts[3])
            return person, session, stance
        except ValueError:
            return None

    def _build_classify_pairs(self):
        """Generate genuine and imposter pairs according to PolyU DBII protocol."""
        # Parse ids into structured dictionary: person -> session -> stance -> id
        parsed = {}
        for img_id in self.data_dict.keys():
            parsed_info = self._parse_id(img_id)
            if not parsed_info:
                continue
            person, session, stance = parsed_info
            parsed.setdefault(person, {}).setdefault(session, {})[stance] = img_id

        genuine_pairs = []
        for person, sessions in parsed.items():
            if 1 in sessions and 2 in sessions:
                s1 = sessions[1]
                s2 = sessions[2]
                for id1 in s1.values():
                    for id2 in s2.values():
                        genuine_pairs.append((id1, id2))

        imposter_pairs = []
        persons = list(parsed.keys())
        for i, pa in enumerate(persons):
            id_a = parsed[pa].get(1, {}).get(1)
            if id_a is None:
                continue
            for pb in persons[i + 1:]:
                id_b = parsed[pb].get(2, {}).get(1)
                if id_b is not None:
                    imposter_pairs.append((id_a, id_b))
                    # also include reverse pairing to cover all A != B combinations
                    id_a2 = parsed[pb].get(1, {}).get(1)
                    id_b2 = parsed[pa].get(2, {}).get(1)
                    if id_a2 is not None and id_b2 is not None:
                        imposter_pairs.append((id_a2, id_b2))

        if self.sets == 'test':
            return genuine_pairs + imposter_pairs

        pair_count = min(len(genuine_pairs), len(imposter_pairs))
        return genuine_pairs[:pair_count] + imposter_pairs[:pair_count]

class PolyUDBIBenchmark(L3SFV2AugmentedBenchmark):
    """Benchmark for the PolyU DBI dataset with classification pair logic."""

    def __init__(self, sets, obj_resize=(512, 512), problem='2GM',
                 filter='intersection', task='match', **args):
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

    # ------------------------------------------------------------------
    # PolyU DBII specific pair generation for classification
    # ------------------------------------------------------------------
    def _parse_id(self, img_id):
        """Parse an image identifier into (person, session, stance).

        Expected format: ``DBII_{person}_{session}_{stance}`` with integer
        components. Returns ``(person, session, stance)`` or ``None`` if the
        pattern does not match.
        """
        parts = img_id.split('_')
        if len(parts) < 4:
            return None
        try:
            person = int(parts[1])
            session = int(parts[2])
            stance = int(parts[3])
            return person, session, stance
        except ValueError:
            return None

    def _build_classify_pairs(self):
        """Generate genuine and imposter pairs according to PolyU DBII protocol."""
        # Parse ids into structured dictionary: person -> session -> stance -> id
        parsed = {}
        for img_id in self.data_dict.keys():
            parsed_info = self._parse_id(img_id)
            if not parsed_info:
                continue
            person, session, stance = parsed_info
            parsed.setdefault(person, {}).setdefault(session, {})[stance] = img_id

        genuine_pairs = []
        for person, sessions in parsed.items():
            if 1 in sessions and 2 in sessions:
                s1 = sessions[1]
                s2 = sessions[2]
                for id1 in s1.values():
                    for id2 in s2.values():
                        genuine_pairs.append((id1, id2))

        imposter_pairs = []
        persons = list(parsed.keys())
        for i, pa in enumerate(persons):
            id_a = parsed[pa].get(1, {}).get(1)
            if id_a is None:
                continue
            for pb in persons[i + 1:]:
                id_b = parsed[pb].get(2, {}).get(1)
                if id_b is not None:
                    imposter_pairs.append((id_a, id_b))
                    # also include reverse pairing to cover all A != B combinations
                    id_a2 = parsed[pb].get(1, {}).get(1)
                    id_b2 = parsed[pa].get(2, {}).get(1)
                    if id_a2 is not None and id_b2 is not None:
                        imposter_pairs.append((id_a2, id_b2))

        if self.sets == 'test':
            return genuine_pairs + imposter_pairs

        pair_count = min(len(genuine_pairs), len(imposter_pairs))
        return genuine_pairs[:pair_count] + imposter_pairs[:pair_count]
