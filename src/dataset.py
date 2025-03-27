import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
import csv
from pathlib import Path
from PIL import Image

class L3SFV2AugmentedDataset:
    def __init__(self, sets, obj_resize=(320, 240), train_root='/green/data/L3SF', test_root=None, val_root=None, cache_path='cache'):
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
        
        # Determine the root directories based on the dataset split.
        self.root_dirs = self._get_root_dirs(sets, train_root, test_root, val_root)
        
        # Collect image files from the provided directories.
        self.image_list = self._collect_images(self.root_dirs)
        
        # Process the images to build the annotation dictionary and save as JSON.
        self.process()

    def _get_root_dirs(self, sets, train_root, test_root, val_root):
        """
        Return a list of Path objects representing the directories to search for images.
        """
        if sets == 'train':
            # Assume training images are organized in subfolders R1–R5.
            # return [Path(os.path.join(train_root, "R6"))]
            return [Path(os.path.join(train_root, f"R{i}")) for i in range(1, 5)]
            return [Path(os.path.join(train_root, "R1"))]
        
        elif sets == 'test':
            return [Path(os.path.join(train_root, "R5"))]
            if test_root is None:
                raise ValueError("For the test set, you must provide a test_root directory.")
            return [Path(test_root)]
        elif sets == 'val':
            return [Path(os.path.join(train_root, "R6"))]
            if val_root is None:
                raise ValueError("For the validation set, you must provide a val_root directory.")
            return [Path(val_root)]
        else:
            raise ValueError("sets must be one of 'train', 'test', or 'val'.")

    def _collect_images(self, root_dirs):
        """
        Walk through the provided directories and collect all JPEG image files.
        """
        images = []
        for dir_path in root_dirs:
            if not dir_path.exists():
                print(f"Directory {dir_path} does not exist; skipping it.")
                continue
            for img_file in dir_path.glob("*.jpg"):
                images.append(img_file)
        return images


    def _get_keypoints(self, img_path):
        """
        Retrieve keypoint annotations for the given image from a csv file.
        Assumes the csv file (located in the same folder as the image) has headers:
        "x", "y", and "id".
        
        Returns:
            A list of dictionaries, each containing:
            - "x": x-coordinate (float)
            - "y": y-coordinate (float)
            - "labels": keypoint label (string) [converted from the "id" field]
        """
        csv_file = img_path.parent / (img_path.stem + '.csv')
        keypoints = []
        if not csv_file.exists():
            print(f"Warning: Keypoint file {csv_file} not found for image {img_path.name}.")
            return keypoints

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    try:
                        x = float(row['x'])
                        y = float(row['y'])
                        # Convert "id" to "labels" for consistency with Pascal VOC
                        kp_label = row['id']
                        keypoints.append({"labels": kp_label, "x": x, "y": y})
                    except Exception as e:
                        print(f"Error reading row in {csv_file}: {e}")
                        continue
        except Exception as e:
            print(f"Error opening {csv_file}: {e}")
        
        return keypoints

    def process(self):
        """
        Process the images to create a JSON annotation file.
        
        The annotation dictionary for each image now includes:
        - "path": full path to the image.
        - "cls": object class (derived from the subject name).
        - "bounds": fixed bounding box [0, 0, 340, 240].
        - "kpts": list of keypoints (each keypoint now has "labels", "x", "y").
        - "univ_size": expected number of keypoints (set to len(kpts)).
        """
        data_dict = {}
        fixed_bounds = [0, 0, 319, 240]  # [xmin, ymin, xmax, ymax]
        
        for img_path in tqdm(self.image_list, desc="Processing images"):
            folder = img_path.parent.name  # e.g., R1, R2, etc.
            file_stem = img_path.stem      # e.g., "1_left_loop_aug_0"
            unique_id = f"{folder}_{file_stem}"
            
            # Parse the filename to extract the subject name and augmentation index.
            parts = file_stem.rsplit('_aug_', 1)
            if len(parts) == 2:
                subject_name, aug_index = parts[0], parts[1]
            else:
                subject_name = file_stem
                aug_index = None
            
            # Retrieve keypoints from the corresponding csv file.
            kpts = self._get_keypoints(img_path)
            
            # Build the annotation dictionary.
            anno = {
                "path": str(img_path),
                "cls": f"{folder}_{subject_name}",         # Use subject name as the object class.
                "bounds": fixed_bounds,
                "kpts": kpts,
                "univ_size": len(kpts)         # Expected number of keypoints.
            }
            
            # (Optional) If you want to also store augmentation info or the folder name:
            anno["augmentation"] = aug_index
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

        This method assumes that for an image file (e.g., "1_left_loop_aug_0.jpg")
        there is a corresponding csv file in the same directory (e.g., "1_left_loop_aug_0.csv")
        that contains keypoint annotations. The csv file is expected to have headers:
        "x", "y", and "id". The returned dictionary mimics the Pascal VOC annotation
        structure.

        The bounding box is computed as [0, 0, min(340, width), min(240, height)],
        ensuring that it does not exceed the original image size.

        Parameters:
            img_path (Path): The path to the image file.

        Returns:
            dict: An annotation dictionary with keys:
                - "path": Full image path (string)
                - "cls": Object class (derived from the subject name in the filename)
                - "bounds": Bounding box [0, 0, x_max, y_max]
                - "kpts": List of keypoint dictionaries (each with "labels", "x", "y")
                - "univ_size": Number of keypoints (int)
                - "augmentation": (Optional) The augmentation index parsed from the filename.
        """
        # Ensure the image exists.
        if not img_path.exists():
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        # Parse the filename to extract subject name and augmentation index.
        file_stem = img_path.stem  # e.g., "1_left_loop_aug_0"
        parts = file_stem.rsplit('_aug_', 1)
        if len(parts) == 2:
            subject_name, aug_index = parts[0], parts[1]
        else:
            subject_name = file_stem
            aug_index = None

        # Open the image to get its dimensions.
        with Image.open(str(img_path)) as img:
            width, height = img.size

        # Compute the bounding box: it should not exceed the image's dimensions.
        xmax = min(340, width)
        ymax = min(240, height)
        bounds = [0, 0, xmax, ymax]

        # Retrieve keypoints from the corresponding csv file.
        keypoints = []
        csv_file = img_path.parent / (img_path.stem + '.csv')
        if csv_file.exists():
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter=',')
                    for row in reader:
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                            kp_label = row['id']
                            keypoints.append({"labels": kp_label, "x": x, "y": y})
                        except Exception as e:
                            print(f"Error parsing a keypoint in {csv_file}: {e}")
                            continue
            except Exception as e:
                print(f"Error reading keypoint file {csv_file}: {e}")
        else:
            print(f"Warning: Keypoint csv file {csv_file} not found for image {img_path.name}.")

        # Build the annotation dictionary.
        anno_dict = {
            "path": str(img_path),
            "cls": subject_name,
            "bounds": bounds,
            "kpts": keypoints,
            "univ_size": len(keypoints)
        }
        
        if aug_index is not None:
            anno_dict["augmentation"] = aug_index

        return anno_dict



# Example usage:
if __name__ == "__main__":
    # For training, images (and their corresponding csv files) are assumed to be in /green/data/L3SF in folders R1–R5.
    dataset_train = L3SFV2AugmentedDataset(
        sets='train',
        obj_resize=(320, 240),
        train_root='/green/data/L3SF_V2/L3SF_V2_Augmented'
    )
    dic = dataset_train._get_anno_dict(Path("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg"))
    print(dic["univ_size"])

