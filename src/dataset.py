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

        if self.task == 'classify':
            self.classify_img_dir = Path('pore-detection/out_of_the_box_detect/Prediction/Pore')
            self.classify_anno_dir = Path('pore-detection/out_of_the_box_detect/Prediction/Coordinates')
        
        # Determine the root directories based on the dataset split.
        self.root_dirs = self._get_root_dirs(sets, train_root, test_root, val_root)
        
        # Collect image files from the provided directories.
        self.image_list = self._collect_images(self.root_dirs)
        
        # Process the images to build the annotation dictionary and save as JSON.
        self.process()

    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """Return a list of Path objects for image search."""
        if self.task == 'classify':
            # Only one directory contains all images for classification
            return [self.classify_img_dir]

        if sets == 'train':
            return [Path(os.path.join(train_root, f"R{i}")) for i in range(1, 4)]
        elif sets == 'test':
            return [Path(os.path.join(train_root, "R4"))]
        elif sets == 'val':
            return [Path(os.path.join(train_root, "R5"))]
        else:
            raise ValueError("sets must be one of 'train', 'test', or 'val'.")

    def _collect_images(self, root_dirs):
        """Collect image files according to the chosen task and split."""
        images = []

        if self.task == 'classify':
            # Filter based on R{num} contained in file name
            if self.sets == 'train':
                allowed = ['R1', 'R2', 'R3']
            elif self.sets == 'test':
                allowed = ['R4']
            elif self.sets == 'val':
                allowed = ['R5']
            else:
                raise ValueError("sets must be one of 'train', 'test', or 'val'.")

            for img_file in root_dirs[0].glob('*.png'):
                stem = img_file.stem
                if any(r in stem for r in allowed):
                    images.append(img_file)
            return images

        for dir_path in root_dirs:
            if not dir_path.exists():
                print(f"Directory {dir_path} does not exist; skipping it.")
                continue
            for img_file in dir_path.glob("*.jpg"):
                images.append(img_file)
        return images


    def _get_keypoints(self, img_path):
        """
        Retrieve keypoint annotations for the given image from a TSV file.
        Assumes the TSV file (located in the same folder as the image) has headers "x" and "y".
        
        Returns:
            A list of dictionaries, each containing:
            - "x": x-coordinate (float)
            - "y": y-coordinate (float)
            - "labels": auto-generated keypoint label (int) that increments
        """
        if self.task == 'classify':
            tsv_file = self.classify_anno_dir / (img_path.stem + '.txt')
        else:
            tsv_file = img_path.parent / (img_path.stem + '.tsv')
        keypoints = []
        if not tsv_file.exists():
            print(f"Warning: Keypoint file {tsv_file} not found for image {img_path.name}.")
            return keypoints

        try:
            with open(tsv_file, 'r') as f:
                if self.task == 'classify':
                    for i, line in enumerate(f):
                        cleaned = line.replace(',', ' ')
                        parts = cleaned.split()
                        if len(parts) < 2:
                            continue
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            keypoints.append({"labels": i, "x": x, "y": y})
                        except Exception as e:
                            print(f"Error reading line in {tsv_file}: {e}")
                else:
                    reader = csv.DictReader(f, delimiter='\t')
                    for i, row in enumerate(reader):
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                            keypoints.append({"labels": i, "x": x, "y": y})
                        except Exception as e:
                            print(f"Error reading row in {tsv_file}: {e}")
                            continue
        except Exception as e:
            print(f"Error opening {tsv_file}: {e}")
        
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
            if self.task == 'classify':
                m = re.search(r'(R[1-5])', img_path.stem)
                folder = m.group(1) if m else 'R0'
            else:
                folder = img_path.parent.name
            file_stem = img_path.stem
            unique_id = f"{folder}_{file_stem}"

            # Retrieve keypoints from the corresponding TSV file.
            kpts = self._get_keypoints(img_path)
            
            with Image.open(str(img_path)) as img:
                width, height = img.size
            
            xmax = max(340, width)
            ymax = max(240, height)
            fixed_bounds = [0, 0, xmax, ymax]
            
            # Build the annotation dictionary.
            anno = {
                "path": str(img_path),
                "cls": f"{unique_id}",
                "bounds": fixed_bounds,
                "kpts": kpts,
                "univ_size": len(kpts)
            }
            
            # Optionally store additional info.
            anno["folder"] = folder
            anno["obj_resize"] = self.obj_resize
            
            data_dict[unique_id] = anno
        
        output_dir = Path("data/L3SFV2Augmented")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.sets}-{self.obj_resize}.json"
        with open(output_file, "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Annotation file saved at: {output_file}")
        
    

    def _get_anno_dict(self, img_path: Path):
        """
        Create an annotation dictionary for a given image.
        
        Assumes that for an image file there is a corresponding TSV file
        (with headers "x" and "y") in the same directory.
        
        The bounding box is computed as [0, 0, min(340, width), min(240, height)],
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
        if self.task == 'classify':
            m = re.search(r'(R[1-5])', img_path.stem)
            folder = m.group(1) if m else 'R0'
        else:
            folder = img_path.parent.name
        subject_name = file_stem
        
        with Image.open(str(img_path)) as img:
            width, height = img.size
        
        
        xmax = min(320, width)
        ymax = min(240, height)
        bounds = [0, 0, xmax, ymax]
        
        keypoints = self._get_keypoints(img_path)
        
        anno_dict = {
            "path": str(img_path),
            "cls": f"{folder}_{subject_name}",
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints)
        }
        
        return anno_dict
        
    


# Example usage:
if __name__ == "__main__":
    # For training, images (and their corresponding csv files) are assumed to be in /green/data/L3SF in folders R1–R5.
    dataset_train = L3SFV2AugmentedDataset(
        sets='train',
        obj_resize=(320, 240),
        train_root='dataset/Synthetic'
    )
    dic = dataset_train._get_anno_dict(Path("dataset/Synthetic/R1/8_right_loop.jpg"))
    print(dic["univ_size"])
