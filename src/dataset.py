"""

The dataset is structured as follows: 
    the images are stored as jpg files in subfolders R1, R2, R3, R4, R5 in "dataset/Pore ground truth/Fingerprint Images"
    the annotations are stored as tsv files in subfolders R1, R2, R3, R4, R5 in "dataset/Pore ground truth/Ground truth"
    
    the image is stored as {subject}.jpg and the respective annotation is stored as {subject}.tsv
    
    the annotations include x y coordinates of the keypoints.
    
Need to update the _get_keypoints(self, img_path): function to get the keypoints from the tsv file. there will no longer be an id column since there will only be 1 instance of a subject. 
This function should now return the x, y coordinates of the tsv file and creates a label column that auto increments by 1

For the process method
You would no longer need to split by "_aug_" since there will only be 1 instance of a specified subject in each folder.
unique id will remain f"{folder}_{file_stem}" as file stems may be repeated across folders but are different subjects
    
For get_anno_dict method you should
    no longer need to split by "_aug_"
    also ensure that the keypoint extraction is not redundant
     
"""

import os
import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from src.gmdataset import RESCALE
from utils.data import train_test_split
from utils.keypoints import subject_pore_labels


class BaseFingerprintDataset(ABC):
    def __init__(self, sets, obj_resize=RESCALE, train_root=None,
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
        # Create cache directory if it doesn't exist yet
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.task = task


        # In-memory cache for keypoints, keyed by (anno_path, mtime)
        self._kpt_cache = {}

        # Determine the root directories based on the dataset split.
        self.root_dirs = self._get_root_dirs(sets, train_root, test_root, val_root)

        # Collect image files from the provided directories.
        self.image_list = self._collect_images(self.root_dirs)

        # Note: processing is deferred. Use `to_json()` to build or reuse output.

    @abstractmethod
    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """Return a list of Path objects for image search."""

    @abstractmethod
    def _get_ids(self, img_path: Path):
        """Return (unique_id, cls_name) for the given image."""

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

    def _get_bounds(self, img_path: Path):
        with Image.open(str(img_path)) as img:
            width, height = img.size
        resize_w, resize_h = self.obj_resize
        xmax = resize_w if width > resize_w else width
        ymax = resize_h if height > resize_h else height
        return [0, 0, xmax, ymax]

    def _augment_anno(self, anno: dict, img_path: Path) -> None:
        """Hook for mixins to add fields to the annotation dictionary."""
        anno["obj_resize"] = self.obj_resize
        return None

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

        get_ids = self._get_ids
        get_keypoints = self._get_keypoints
        get_bounds = self._get_bounds
        augment = self._augment_anno

        for img_path in self.image_list:
            unique_id, cls_name = get_ids(img_path)
            kpts = get_keypoints(img_path)
            bounds = get_bounds(img_path)

            anno = {
                "path": str(img_path),
                "cls": cls_name,
                "bounds": bounds,
                "kpts": kpts,
                "univ_size": len(kpts)
            }

            augment(anno, img_path)
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
        
        unique_id, cls_name = self._get_ids(img_path)
        bounds = self._get_bounds(img_path)
        keypoints = self._get_keypoints(img_path)

        anno_dict = {
            "path": str(img_path),
            "cls": cls_name,
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints)
        }

        self._augment_anno(anno_dict, img_path)
        return anno_dict

class KeypointsForMultiSessionMixin:
    """
    Mixin to handle datasets where the same subject may have multiple sessions/images,
    and keypoints need to be unified across those sessions.
    """
    def __init__(self, *args, annotation_path, **kwargs):
        self.annotation_path = Path(annotation_path)
        super().__init__(*args, **kwargs)

    def _get_anno_by_subject(self, anno_path: Path):
        subject_dict = {}
        with open(anno_path, 'r') as f:
            for line in f:
                anno = json.loads(line)
                subject = anno['subject']

                if subject_dict.get(subject) is not None:
                    subject_dict[subject].append(anno)
                else:
                    subject_dict[subject] = [anno]

        return subject_dict

    def _get_keypoint_data(self):
        anno_path = self.annotation_path
        try:
            mtime = anno_path.stat().st_mtime
        except OSError:
            mtime = None

        cache_key = (str(anno_path), mtime)
        cached = self._kpt_cache.get(cache_key)
        if cached is None:
            subject_dict = self._get_anno_by_subject(anno_path)
            cached = subject_pore_labels(subject_dict)
            self._kpt_cache.clear()
            self._kpt_cache[cache_key] = cached
        return cached

    def _build_keypoint_image_name(self, img_path: Path):
        return img_path.stem

    def _build_keypoints(self, image_name, global_label, kp_desc, kps):
        keypoints = []
        desc = kp_desc.get(image_name)
        if desc is None:
            return keypoints

        for kp in kps.get(image_name, ()):
            label = desc.get_keypoint(kp)
            if global_label.get(label) is not None:
                label = global_label[label]
            keypoints.append({"labels": label, "x": kp[1], "y": kp[0]})

        return keypoints

    def _get_keypoints(self, img_path):
        image = self._build_keypoint_image_name(img_path)
        global_label, kp_desc, kps = self._get_keypoint_data()
        return self._build_keypoints(image, global_label, kp_desc, kps)

class KeypointsFromAnnotationMixin:
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
        possible_exts = ('.tsv', '.csv', '.txt')
        parent = img_path.parent
        stem = img_path.stem

        anno_file = None
        delimiter = None
        ext_used = None

        for ext in possible_exts:
            candidate = parent / (stem + ext)
            if candidate.exists():
                anno_file = candidate
                delimiter = '\t' if ext == '.tsv' else ','
                ext_used = ext
                break

        if not anno_file:
            print(f"Warning: Keypoint file not found for image {img_path.name}.")
            return []

        try:
            mtime = os.path.getmtime(anno_file)
        except OSError:
            mtime = None
        cache_key = (str(anno_file), mtime)
        cached = self._kpt_cache.get(cache_key)
        if cached is not None:
            return cached

        keypoints = []
        prefix = f"{parent.name}_{stem}"
        append = keypoints.append

        try:
            if ext_used == '.txt':
                with open(anno_file, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) < 2:
                            continue
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                        except ValueError:
                            continue
                        append({"labels": f"{prefix}_{i}", "x": x, "y": y})
            else:
                with open(anno_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for i, row in enumerate(reader):
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                        except (KeyError, TypeError, ValueError):
                            continue
                        append({"labels": f"{prefix}_{i}", "x": x, "y": y})
        except Exception as e:
            print(f"Error opening {anno_file}: {e}")
        else:
            self._kpt_cache[cache_key] = keypoints

        return keypoints


class RootDirByRFolderMixin:
    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        if sets == 'train':
            return [Path(os.path.join(train_root, f"R{i}")) for i in range(1, 4)]
        if sets == 'test':
            return [Path(os.path.join(train_root, "R4"))]
        if sets == 'val':
            return [Path(os.path.join(train_root, "R5"))]
        raise ValueError("sets must be one of 'train', 'test', or 'val'.")


class RootDirTrainTestValMixin:
    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        if sets == 'train':
            return [Path(os.path.join(train_root, "train"))]
        if sets == 'test':
            return [Path(os.path.join(train_root, "test"))]
        if sets == 'val':
            return [Path(os.path.join(train_root, "val"))]
        raise ValueError("sets must be one of 'train', 'test', or 'val'.")


class FolderStemIdMixin:
    def _get_ids(self, img_path: Path):
        """ Ids are the entire file name and folder path since there is only one instance per subject """
        folder = img_path.parent.name
        file_stem = img_path.stem
        parts = file_stem.split('_')
        if len(parts) >= 2:
            cls_name = f"{parts[0]}"
        else:
            cls_name = file_stem
        unique_id = f"{folder}_{cls_name}"
        cls_name = unique_id
        return unique_id, cls_name


class StemPartsIdMixin:
    def _get_ids(self, img_path: Path):
        """ Ids are the first two parts of the file stem joined by an underscore """
        file_stem = img_path.stem
        parts = file_stem.split('_')
        if len(parts) >= 2:
            cls_name = f"{parts[0]}"
        else:
            cls_name = file_stem
        unique_id = file_stem
        return unique_id, cls_name



class FolderFieldMixin:
    def _augment_anno(self, anno: dict, img_path: Path) -> None:
        anno["folder"] = img_path.parent.name
        return super()._augment_anno(anno, img_path)


class L3SFV2AugmentedDataset(
    FolderStemIdMixin,
    FolderFieldMixin,
    RootDirByRFolderMixin,
    KeypointsFromAnnotationMixin,
    BaseFingerprintDataset,
):
    def __init__(self, sets, obj_resize=RESCALE, train_root='dataset/Synthetic',
                 test_root=None, val_root=None, cache_path='cache', task='classify'):
        self.output_dir =  Path("data/L3SFV2AugmentedDataset")
        super().__init__(sets, obj_resize, train_root, test_root, val_root, cache_path, task)   

class PolyUDBII(
    StemPartsIdMixin,
    RootDirTrainTestValMixin,
    KeypointsForMultiSessionMixin,
    BaseFingerprintDataset,
):
    def __init__(self, sets, obj_resize=RESCALE, train_root='dataset/PolyU/DBII',
                 test_root=None, val_root=None, cache_path='cache', task='match',
                 annotation_path='dataset/polyU/annotations'):
        self.output_dir = Path("data/PolyU-DBII")
        if not os.path.exists(train_root):
            train_test_split(train_root, img_path='dataset/polyU/DBII')
        super().__init__(
            sets,
            obj_resize,
            train_root,
            test_root,
            val_root,
            cache_path,
            task,
            annotation_path=annotation_path,
        )


class PolyUDBI(PolyUDBII):
    """
    PolyUDBI dataset class, inheriting from PolyUDBII.
    This class can be used to handle the PolyUDBI dataset with similar functionality.
    """
    def __init__(self, sets, obj_resize=RESCALE, train_root='dataset/PolyU/DBI',
                 test_root=None, val_root=None, cache_path='cache', task='match'):
        self.output_dir = Path("data/PolyU-DBI")
        super().__init__(sets, obj_resize, train_root, test_root, val_root, cache_path, task)
        
class L3SF(
    FolderStemIdMixin,
    FolderFieldMixin,
    RootDirByRFolderMixin,
    KeypointsForMultiSessionMixin,
    BaseFingerprintDataset,
):
    def __init__(self, sets, obj_resize=RESCALE, train_root='dataset/L3SF_V2/L3-SF',
                 test_root=None, val_root=None, cache_path='cache', task='match',
                 annotation_path='dataset/L3SF_V2/results/alignment_annotations.jsonl'):
        self.output_dir = Path("data/L3-SF")
        super().__init__(
            sets,
            obj_resize,
            train_root,
            test_root,
            val_root,
            cache_path,
            task,
            annotation_path=annotation_path,
        )

    def _collect_images(self, root_dirs):
        images = []
        for dir_path in root_dirs:
            if not dir_path.exists():
                print(f"Directory {dir_path} does not exist; skipping it.")
                continue
            for ext in ("*.jpg", "*.png"):
                images.extend(dir_path.glob(ext))
        return images

    def _build_keypoint_image_name(self, img_path: Path):
        return f"{img_path.parent.name}_{img_path.stem}"

    def _get_ids(self, img_path: Path):
        folder = img_path.parent.name
        file_stem = img_path.stem
        parts = file_stem.split('_')
        person = parts[0] if parts else file_stem
        unique_id = f"{folder}_{file_stem}"
        cls_name = f"{folder}_{person}"
        return unique_id, cls_name


if __name__ == "__main__":
    # For training, images (and their corresponding csv files) are assumed to be in /green/data/L3SF in folders R1–R5.
    dataset_train = PolyUDBII(
        sets='train',
        obj_resize=RESCALE,
    )
    # Build or reuse the processed JSON annotations file
    json_path = dataset_train.to_json()
    print(f"Annotations JSON: {json_path}")


# if __name__ == "__main__":
#     from pathlib import Path

#     class DebugPolyU(KeypointsForMultiSessionMixin, BaseFingerprintDataset):
#         def _get_root_dirs(self, sets, train_root, test_root, val_root):
#             return []

#         def _get_ids(self, img_path: Path):
#             return img_path.stem, img_path.stem

#         def __init__(self):
#             self.output_dir = Path("tmp")
#             super().__init__(sets="train", train_root="dataset/PolyU/DBII")

#     ds = DebugPolyU()
#     img_path = Path("dataset/PolyU/DBII/train/1_2_5.jpg")
#     kpts = ds._get_keypoints(img_path)

#     for kp in kpts:
#         print(f"label={kp['labels']} x={kp['x']} y={kp['y']}")

