import os
import tempfile
import json
from pathlib import Path
from pygmtools.benchmark import Benchmark  # Import the original Benchmark class
# Import your new dataset class.
# (Adjust the import according to where your L3SFV2AugmentedDataset is defined.)
from src.dataset import L3SFV2AugmentedDataset
from PIL import Image

class L3SFV2AugmentedBenchmark(Benchmark):
    """
    A Benchmark subclass for our new L3SFV2Augmented dataset.
    
    This class inherits from the Benchmark base class but overrides its
    initialization so that it uses our new dataset. (We ignore the eval and eval_cls
    methods since they are not of interest.)
    """
    def __init__(self, sets, obj_resize=(320, 240), problem='2GM', filter='intersection', **args):
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
            dataset_instance.suffix = "(320, 240)"  # you can choose any suffix you like
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
    
