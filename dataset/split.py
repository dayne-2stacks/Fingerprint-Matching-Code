
import argparse
"""
split.py
Splits a fingerprint dataset into training, validation, and test sets by grouping images according to person identifiers.
Copies both fingerprint images and their corresponding annotation files (e.g., detected pores) into separate split directories.
When to use:
    - Use this script after running pore detection and saving the resulting annotation files.
    - Intended for organizing datasets before training or evaluating machine learning models on fingerprint data.
How to use:
    1. Ensure your dataset directory contains:
        - A 'Fingerprint' subdirectory with fingerprint image files.
        - A 'Coordinates' subdirectory with annotation files (e.g., pore coordinates) named to match the images.
    2. Run the script from the command line:
        python split.py --db-root /path/to/your/dataset
    3. The script will create 'train', 'val', and 'test' directories inside the dataset root, each containing the corresponding images and annotation files.
Details:
    - Images are grouped by person based on the filename convention (expects person identifier as the second underscore-separated part).
    - Persons are randomly shuffled and split: 60% for training, 20% for validation, and 20% for testing.
    - Both images and their annotation files are copied (not moved) to the split directories.
    - Existing split directories will be reused or created if missing.
Arguments:
    --db-root: Path to the root directory of the fingerprint database (default: ./PolyU)
"""
import os
import shutil
import random
from collections import defaultdict

def get_parser():
    parser = argparse.ArgumentParser(description="Split fingerprint dataset")
    parser.add_argument(
        "--db-root",
        type=str,
        default="./PolyU",
        help="Path to the root directory of the fingerprint database (default: ./PolyU)"
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f"Database root: {args.db_root}")
    
    annotation_path = os.path.join(args.db_root, "Coordinates")
    image_path = os.path.join(args.db_root, "Fingerprint")
    print(f"Annotation path: {annotation_path}")
    print(f"Image path: {image_path}")
    
    # Collect all image files
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    # Group images by person (leading number)
    person_to_images = defaultdict(list)
    for fname in image_files:
        parts = fname.split('_')
        if len(parts) < 2:
            continue  # skip malformed names
        person = parts[1]
        person_to_images[person].append(fname)

    # Shuffle persons and split into train/val/test
    persons = list(person_to_images.keys())
    random.shuffle(persons)
    n = len(persons)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_persons = persons[:n_train]
    val_persons = persons[n_train:n_train + n_val]
    test_persons = persons[n_train + n_val:]

    splits = {
        'train': train_persons,
        'val': val_persons,
        'test': test_persons
    }

    # Create split folders (images and annotations together)
    for split in splits:
        split_dir = os.path.join(args.db_root, split)
        os.makedirs(split_dir, exist_ok=True)

    # Prepare annotation file lookup for efficiency
    annotation_files = set(os.listdir(annotation_path))

    # Move/copy images and annotations to split folders
    for split, persons in splits.items():
        split_dir = os.path.join(args.db_root, split)
        for person in persons:
            for fname in person_to_images[person]:
                # Copy image
                src_img = os.path.join(image_path, fname)
                dst_img = os.path.join(split_dir, fname)
                shutil.copy2(src_img, dst_img)
                # Copy annotation (if exists)
                ann_name = os.path.splitext(fname)[0] + '.txt'
                if ann_name in annotation_files:
                    src_ann = os.path.join(annotation_path, ann_name)
                    dst_ann = os.path.join(split_dir, ann_name)
                    shutil.copy2(src_ann, dst_ann)

    print("Dataset split complete.")