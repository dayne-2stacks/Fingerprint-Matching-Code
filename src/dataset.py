"""

The dataset is structured as follows: 
    the images are stored as jpg files in subfolders R1, R2, R3, R4, R5 in "dataset/Pore ground truth/Fingerprint Images"
    the annotations are stored as tsv files in subfolders R1, R2, R3, R4, R5 in "dataset/Pore ground truth/Ground truth"
    
    the image is stored as {subject}.jpg and the respective annotation is stored as {subject}.tsv
    
    the annotations include x y coordinates of the keypoints.
    
You should update the _get_keypoints(self, img_path): function to get the keypoints from the tsv file. there will no longer be an id column since there will only be 1 instance of a subject. 
This function should now return the x, y coordinates of the tsv file and creates a label column that auto increments by 1

For the process method
You would no longer need to split by "_aug_" since there will only be 1 instance of a specified subject in each folder.
unique id will remain f"{folder}_{file_stem}" as file stems may be repeated across folders but are different subjects
    
For get_anno_dict method you should
    similarly, no longer need to split by "_aug_"
    also ensure that the keypoint extraction is not redundant
     
"""


import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
import csv
from pathlib import Path
from PIL import Image
import re

class L3SFV2AugmentedDataset:
    def __init__(self, sets, obj_resize=(512, 512), train_root='dataset/Synthetic',
                 test_root=None, val_root=None, cache_path='cache', task='match'):
        """
        Initialize the dataset.
        
        :param sets: str, one of 'train', 'test', or 'val'
        :param obj_resize: tuple, e.g. (width, height) for resizing images
        :param train_root: str, root directory for training images (which has subfolders R1–R5)
        :param test_root: str, root directory for test images (if sets=='test')
        :param val_root: str, root directory for validation images (if sets=='val')
        :param cache_path: str, directory to cache any generated files (optional)
        """        
        self.sets = sets
        self.obj_resize = obj_resize
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.task = task

        # Ensure an output directory is available. Subclasses can set
        # self.output_dir before calling super().__init__ to override this.
        if not hasattr(self, "output_dir"):
            self.output_dir = Path("data/L3SFV2AugmentedDataset")

        # In-memory cache for keypoints, keyed by (anno_path, mtime)
        self._kpt_cache = {}

        # Determine the root directories based on the dataset split.
        self.root_dirs = self._get_root_dirs(sets, train_root, test_root, val_root)
        
        # Collect image files from the provided directories.
        self.image_list = self._collect_images(self.root_dirs)

        # Note: processing is deferred. Use `to_json()` to build or reuse output.

    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """Return a list of Path objects for image search."""
        if sets == 'train':
            return [Path(os.path.join(train_root, f"R{i}")) for i in range(1, 4)]
        elif sets == 'test':
            return [Path(os.path.join(train_root, "R4"))]
        elif sets == 'val':
            return [Path(os.path.join(train_root, "R5"))]
        else:
            raise ValueError("sets must be one of 'train', 'test', or 'val'.")

    def _collect_images(self, root_dirs):
        """Collect image files (jpg or png) according to the chosen task and split."""
        images = []

        for dir_path in root_dirs:
            if not dir_path.exists():
                print(f"Directory {dir_path} does not exist; skipping it.")
                continue
            for ext in ("*.jpg", "*.png"):
                for img_file in dir_path.glob(ext):
                    images.append(img_file)
        return images

    def _output_file_path(self) -> Path:
        """Return the expected path to the processed JSON for this instance."""
        return Path(self.output_dir) / f"{self.sets}-{self.obj_resize}.json"

    def to_json(self, force: bool = False) -> Path:
        """
        Build annotations JSON if needed and return its path.
        - If the JSON already exists and force=False, reuse the existing file.
        - If force=True, rebuild the JSON.
        """
        output_file = self._output_file_path()
        if output_file.exists() and not force:
            print(f"Using existing annotation file: {output_file}")
            return output_file
        # (Re)build annotations
        self.process()
        return output_file

    def clear(self) -> None:
        """Delete the processed JSON file for this dataset instance, if it exists."""
        output_file = self._output_file_path()
        try:
            if output_file.exists():
                output_file.unlink()
                print(f"Deleted annotation file: {output_file}")
            else:
                print(f"No annotation file to delete: {output_file}")
        except Exception as e:
            print(f"Failed to delete {output_file}: {e}")


    def _get_keypoints(self, img_path):
        """
        Retrieve keypoint annotations for the given image from a TSV, CSV, or TXT file.
        - TSV: tab-delimited, with header row ("x", "y")
        - CSV: comma-delimited, with header row ("x", "y")
        - TXT: comma-delimited, no header row, just x,y per line

        Returns:
            A list of dictionaries, each containing:
            - "x": x-coordinate (float)
            - "y": y-coordinate (float)
            - "labels": unique keypoint label constructed as
              ``{folder}_{file_stem}_{index}``
        """
        possible_exts = ['.tsv', '.csv', '.txt']
        anno_file = None
        delimiter = None
        ext_used = None

        for ext in possible_exts:
            candidate = img_path.parent / (img_path.stem + ext)
            if candidate.exists():
                anno_file = candidate
                delimiter = '\t' if ext == '.tsv' else ','
                ext_used = ext
                break

        if not anno_file:
            print(f"Warning: Keypoint file not found for image {img_path.name}.")
            return []

        # Check cache by (anno_path, mtime)
        try:
            mtime = os.path.getmtime(anno_file)
        except OSError:
            mtime = None
        cache_key = (str(anno_file), mtime)
        if cache_key in self._kpt_cache:
            return self._kpt_cache[cache_key]

        keypoints = []

        prefix = f"{img_path.parent.name}_{img_path.stem}"
        try:
            if ext_used == '.txt':
                with open(anno_file, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            x_str, y_str = line.split(',')
                            x = float(x_str)
                            y = float(y_str)
                            label = f"{prefix}_{i}"
                            keypoints.append({"labels": label, "x": x, "y": y})
                        except Exception as e:
                            print(f"Error reading line in {anno_file}: {e}")
                            continue
            else:
                with open(anno_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for i, row in enumerate(reader):
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                            label = f"{prefix}_{i}"
                            keypoints.append({"labels": label, "x": x, "y": y})
                        except Exception as e:
                            print(f"Error reading row in {anno_file}: {e}")
                            continue
        except Exception as e:
            print(f"Error opening {anno_file}: {e}")
        else:
            # Save to cache only if parsing succeeded without raising.
            self._kpt_cache[cache_key] = keypoints

        return keypoints
        

    def process(self):
        """
        Process the images to create a JSON annotation file.
        
        The annotation dictionary for each image includes:
        - "path": full path to the image.
        - "cls": subject name formed by the folder and file stem joined by an underscore.
        - "bounds": fixed bounding box [0, 0, 319, 240].
        - "kpts": list of keypoints (each with "labels", "x", "y").
        - "univ_size": number of keypoints.
        """
            
        data_dict = {}
        
        
        for img_path in tqdm(self.image_list, desc="Processing images"):
            folder = img_path.parent.name
            file_stem = img_path.stem
            unique_id = f"{folder}_{file_stem}"
            cls_name = unique_id

            # Retrieve keypoints from the corresponding TSV file.
            kpts = self._get_keypoints(img_path)
            
            with Image.open(str(img_path)) as img:
                width, height = img.size
            
            xmax = min(320, width)
            ymax = min(240, height)
            fixed_bounds = [0, 0, xmax, ymax]
            
            # Build the annotation dictionary.
            anno = {
                "path": str(img_path),
                "cls": cls_name,
                "bounds": fixed_bounds,
                "kpts": kpts,
                "univ_size": len(kpts)
            }
            
            # Optionally store additional info.
            anno["folder"] = folder
            anno["obj_resize"] = self.obj_resize
            
            data_dict[unique_id] = anno
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = self._output_file_path()
        with open(output_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Annotation file saved at: {output_file}")
        
    

    def _get_anno_dict(self, img_path: Path):
        """
        Create an annotation dictionary for a given image.
        
        Assumes that for an image file there is a corresponding TSV file
        (with headers "x" and "y") in the same directory.
        
        The bounding box is computed as [0, 0, min(320, width), min(240, height)],
        ensuring that it does not exceed the original image size.
        
        Returns:
            dict: An annotation dictionary with keys:
                - "path": full image path (string)
                - "cls": subject name (folder and file stem joined by "_")
                - "bounds": bounding box [0, 0, x_max, y_max]
                - "kpts": list of keypoint dictionaries (each with "labels", "x", "y")
                - "univ_size": number of keypoints (int)
        """
        if not img_path.exists():
            raise FileNotFoundError(f"Image file {img_path} does not exist.")
        
        file_stem = img_path.stem
        folder = img_path.parent.name
        unique_id = f"{folder}_{file_stem}"
        cls_name = unique_id
        subject_name = file_stem
        
        with Image.open(str(img_path)) as img:
            width, height = img.size
        
        
        xmax = min(320, width)
        ymax = min(240, height)
        bounds = [0, 0, xmax, ymax]
        
        keypoints = self._get_keypoints(img_path)
        
        anno_dict = {
            "path": str(img_path),
            "cls": cls_name,
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints)
        }
        
        return anno_dict


class PolyUDBII(L3SFV2AugmentedDataset):
    def __init__(self, sets, obj_resize=(512, 512), train_root='dataset/PolyU/DBII',
                 test_root=None, val_root=None, cache_path='cache', task='match'):
        self.output_dir = Path("data/PolyU-DBII")
        super().__init__(sets, obj_resize, train_root, test_root, val_root, cache_path, task)
        
    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """Return a list of Path objects for image search."""
        if sets == 'train':
            return [Path(os.path.join(train_root, "train"))]
        elif sets == 'test':
            return [Path(os.path.join(train_root, "test"))]
        elif sets == 'val':
            return [Path(os.path.join(train_root, "val"))]
        else:
            raise ValueError("sets must be one of 'train', 'test', or 'val'.")
        
    def process(self):
        """
        Process the images to create a JSON annotation file.
        
        The annotation dictionary for each image includes:
        - "path": full path to the image.
        - "cls": subject name formed by the folder and file stem joined by an underscore.
        - "bounds": fixed bounding box [0, 0, 319, 240].
        - "kpts": list of keypoints (each with "labels", "x", "y").
        - "univ_size": number of keypoints.
        """
            
        data_dict = {}
        
        # PolyUDBII: cls_name is {dbname}_{id} from {dbname}_{id}_{session}_{position}
        # unique_id is the full stem: {dbname}_{id}_{session}_{position}
        for img_path in self.image_list:
            file_stem = img_path.stem  # e.g., DBII_001_01_01
            parts = file_stem.split('_')
            if len(parts) >= 2:
                cls_name = f"{parts[0]}_{parts[1]}"
            else:
                cls_name = file_stem  # fallback if unexpected format
            unique_id = file_stem
        

            # Retrieve keypoints from the corresponding TSV file.
            kpts = self._get_keypoints(img_path)
            
            with Image.open(str(img_path)) as img:
                width, height = img.size
            
            xmax = min(320, width)
            ymax = min(240, height)
            fixed_bounds = [0, 0, xmax, ymax]
            
            # Build the annotation dictionary.
            anno = {
                "path": str(img_path),
                "cls": cls_name,
                "bounds": fixed_bounds,
                "kpts": kpts,
                "univ_size": len(kpts)
            }
            
            # Optionally store additional info.
            anno["obj_resize"] = self.obj_resize
            
            data_dict[unique_id] = anno
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.sets}-{self.obj_resize}.json"
        with open(output_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Annotation file saved at: {output_file}")
        
    def _get_anno_dict(self, img_path: Path):
        """
        Create an annotation dictionary for a given image, following the logic of PolyUDBII.process.
        
        The class name ("cls") is constructed as {dbname}_{id} from the file stem {dbname}_{id}_{session}_{position}.
        The unique_id is the full file stem.
        The bounding box is [0, 0, min(320, width), min(240, height)].
        """
        if not img_path.exists():
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        file_stem = img_path.stem  # e.g., DBII_001_01_01
        parts = file_stem.split('_')
        if len(parts) >= 2:
            cls_name = f"{parts[0]}_{parts[1]}"
        else:
            cls_name = file_stem  # fallback if unexpected format
        unique_id = file_stem

        with Image.open(str(img_path)) as img:
            width, height = img.size

        xmax = min(320, width)
        ymax = min(240, height)
        bounds = [0, 0, xmax, ymax]

        keypoints = self._get_keypoints(img_path)

        anno_dict = {
            "path": str(img_path),
            "cls": cls_name,
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints),
            "obj_resize": self.obj_resize
        }

        return anno_dict
       
class PolyUDBI(PolyUDBII):
    """
    PolyUDBI dataset class, inheriting from PolyUDBII.
    This class can be used to handle the PolyUDBI dataset with similar functionality.
    """
    def __init__(self, sets, obj_resize=(512, 512), train_root='dataset/PolyU/DBI',
                 test_root=None, val_root=None, cache_path='cache', task='match'):
        self.output_dir = Path("data/PolyU-DBI")
        super().__init__(sets, obj_resize, train_root, test_root, val_root, cache_path, task)
        
class L3SF(L3SFV2AugmentedDataset):
    def __init__(self, sets, obj_resize=(512, 512), train_root='dataset/L3-SF',
                 test_root=None, val_root=None, cache_path='cache', task='match'):
        self.output_dir = Path("data/L3-SF")
        super().__init__(sets, obj_resize, train_root, test_root, val_root, cache_path, task)
        
    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """Return a list of Path objects for image search."""
        if sets == 'train':
            return [Path(os.path.join(train_root, "train"))]
        elif sets == 'test':
            return [Path(os.path.join(train_root, "test"))]
        elif sets == 'val':
            return [Path(os.path.join(train_root, "val"))]
        else:
            raise ValueError("sets must be one of 'train', 'test', or 'val'.")
        
    def process(self):
        """
        Process the images to create a JSON annotation file.
        
        The annotation dictionary for each image includes:
        - "path": full path to the image.
        - "cls": subject name formed by the folder and file stem joined by an underscore.
        - "bounds": fixed bounding box [0, 0, 319, 240].
        - "kpts": list of keypoints (each with "labels", "x", "y").
        - "univ_size": number of keypoints.
        """
            
        data_dict = {}
        
        # PolyUDBII: cls_name is {dbname}_{id} from {dbname}_{id}_{session}_{position}
        # unique_id is the full stem: {dbname}_{id}_{session}_{position}
        for img_path in self.image_list:
            file_stem = img_path.stem  # e.g., DBII_001_01_01
            parts = file_stem.split('_')
            if len(parts) >= 2:
                cls_name = f"{parts[0]}_{parts[1]}"
            else:
                cls_name = file_stem  # fallback if unexpected format
            unique_id = file_stem
        

            # Retrieve keypoints from the corresponding TSV file.
            kpts = self._get_keypoints(img_path)
            
            with Image.open(str(img_path)) as img:
                width, height = img.size
            
            xmax = min(320, width)
            ymax = min(240, height)
            fixed_bounds = [0, 0, xmax, ymax]
            
            # Build the annotation dictionary.
            anno = {
                "path": str(img_path),
                "cls": cls_name,
                "bounds": fixed_bounds,
                "kpts": kpts,
                "univ_size": len(kpts)
            }
            
            # Optionally store additional info.
            anno["obj_resize"] = self.obj_resize
            
            data_dict[unique_id] = anno
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.sets}-{self.obj_resize}.json"
        with open(output_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Annotation file saved at: {output_file}")
        
    def _get_anno_dict(self, img_path: Path):
        """
        Create an annotation dictionary for a given image, following the logic of PolyUDBII.process.
        
        The class name ("cls") is constructed as {dbname}_{id} from the file stem {dbname}_{id}_{session}_{position}.
        The unique_id is the full file stem.
        The bounding box is [0, 0, min(320, width), min(240, height)].
        """
        if not img_path.exists():
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        file_stem = img_path.stem  # e.g., DBII_001_01_01
        parts = file_stem.split('_')
        if len(parts) >= 2:
            cls_name = f"{parts[0]}_{parts[1]}"
        else:
            cls_name = file_stem  # fallback if unexpected format
        unique_id = file_stem

        with Image.open(str(img_path)) as img:
            width, height = img.size

        xmax = min(320, width)
        ymax = min(240, height)
        bounds = [0, 0, xmax, ymax]

        keypoints = self._get_keypoints(img_path)

        anno_dict = {
            "path": str(img_path),
            "cls": cls_name,
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints),
            "obj_resize": self.obj_resize
        }

        return anno_dict
       
        
# Example usage:
if __name__ == "__main__":
    # For training, images (and their corresponding csv files) are assumed to be in /green/data/L3SF in folders R1–R5.
    dataset_train = PolyUDBII(
        sets='train',
        obj_resize=(320, 240),
    )
    # Build or reuse the processed JSON annotations file
    json_path = dataset_train.to_json()
    print(f"Annotations JSON: {json_path}")
