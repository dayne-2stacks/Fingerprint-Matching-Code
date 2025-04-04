import os
import tempfile
import json
from pathlib import Path
from pygmtools.benchmark import Benchmark  # Import the original Benchmark class
# Import your new dataset class.
# (Adjust the import according to where your L3SFV2AugmentedDataset is defined.)
from src.dataset import L3SFV2AugmentedDataset
from PIL import Image
import random
import itertools
import numpy as np
from scipy.sparse import coo_matrix

class L3SFV2AugmentedBenchmark(Benchmark):
    """
    A Benchmark subclass for our new L3SFV2Augmented dataset.
    
    This class inherits from the Benchmark base class but overrides its
    initialization so that it uses our new dataset. (We ignore the eval and eval_cls
    methods since they are not of interest.)
    """
    def __init__(self, sets, obj_resize=(512, 512), problem='2GM', filter='intersection', **args):
        # Instead of a dataset name from a fixed list, we use our new dataset.
        self.name = "L3SFV2Augmented"  # our custom dataset name
        self.problem = problem
        self.filter = filter
        self.sets = sets
        self.obj_resize = obj_resize

        # Instantiate our new dataset.
        # (The dataset __init__ is expected to generate the necessary JSON files.)
        dataset_instance = L3SFV2AugmentedDataset(sets, obj_resize, **args)

        # Make sure that the dataset has the attributes that Benchmark expects.
        # For example, Benchmark later uses:
        #   - dataset_dir: a directory where JSON annotation files are saved.
        #   - suffix: a string suffix used in the JSON filenames.
        #   - classes: a list of available object classes.
        #
        # If your dataset class does not already define these, set them here:
        if not hasattr(dataset_instance, "dataset_dir"):
            dataset_instance.dataset_dir = "data/L3SFV2Augmented"
        if not hasattr(dataset_instance, "suffix"):
            dataset_instance.suffix = f"{obj_resize}"  # you can choose any suffix you like
        if not hasattr(dataset_instance, "classes"):
            
            # As an example, derive the list of classes by reading the JSON file.
            json_file = os.path.join(dataset_instance.dataset_dir, f"{sets}-{dataset_instance.suffix}.json")
            with open(json_file, "r") as f:
                data_dict = json.load(f)
            
            dataset_instance.classes = list({data_dict[k]['cls'] for k in data_dict})
            
            
    
        self.classes = dataset_instance.classes

        # Build the paths for the unified data interface.
        self.data_path = json_file
        self.data_list_path = json_file
        

        # Load the data dictionary from the JSON file.
        with open(self.data_path, "r") as f:
            self.data_dict = json.load(f)
            
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
        data_list = []
        for keys in ids:
            obj_dict = dict()
            boundbox = self.data_dict[keys]['bounds']
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

# ------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Suppose your training images (and corresponding TSV annotation files)
    # are under '/green/data/L3SF_V2/L3SF_V2_Augmented'
    benchmark_instance = L3SFV2AugmentedBenchmark(
        sets='train',
        obj_resize=(320, 240),
        train_root='/green/data/L3SF_V2/L3SF_V2_Augmented'
    )

    # Now you can use benchmark_instance.get_data, benchmark_instance.rand_get_data,
    # or any other methods defined in Benchmark. Since you are not interested in evaluation,
    # you can simply ignore eval/eval_cls.
    # print("Dataset classes:", benchmark_instance.classes)
    with Image.open(str("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/1_left_loop_aug_0.jpg")) as img:
        print(img.size)
    print("Data path:", benchmark_instance.obj_resize)
    data_dict, perm, ids = benchmark_instance.get_data(["R1_8_right_loop_aug_0", "R1_8_right_loop_aug_1"])
    print(data_dict[1]["univ_size"])
    print(perm)
    
