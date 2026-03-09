import os
import shutil
from pathlib import Path

def combine_items(image_root, ann_root, target_root, folders):
    """
    For each folder in folders, copy the jpg images from image_root and
    the corresponding tsv files from ann_root into the same folder under target_root.
    """
    for folder in folders:
        folder_img = image_root / folder
        folder_ann = ann_root / folder
        target_folder = target_root / folder
        target_folder.mkdir(parents=True, exist_ok=True)
        print(f"Processing folder: {folder}")
        
        # Process each image in the current folder.
        for img_file in folder_img.glob("*.jpg"):
            file_stem = img_file.stem
            ann_file = folder_ann / (file_stem + ".tsv")
            
            # Copy image file
            try:
                shutil.copy2(img_file, target_folder)
            except Exception as e:
                print(f"Error copying image {img_file}: {e}")
            
            # Copy annotation file if it exists
            if ann_file.exists():
                try:
                    shutil.copy2(ann_file, target_folder)
                except Exception as e:
                    print(f"Error copying annotation {ann_file}: {e}")
            else:
                print(f"Warning: Annotation file {ann_file} not found for image {img_file.name}")

def main():
    # Define source directories.
    image_root = Path("Pore ground truth/Fingerprint Images")
    ann_root = Path("Pore ground truth/Ground truth")
    
    # Define target directory.
    target_root = Path("Synthetic")
    target_root.mkdir(parents=True, exist_ok=True)
    
    # List of R folders to process.
    folders = [f"R{i}" for i in range(1, 6)]
    
    combine_items(image_root, ann_root, target_root, folders)
    print("Combination complete. Check the Synthetic folder.")

if __name__ == "__main__":
    main()
