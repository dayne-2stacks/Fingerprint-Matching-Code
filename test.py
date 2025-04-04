#!/usr/bin/env python
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch_geometric as pyg
from utils.augmentation import augment_image
from utils.build_graphs import build_graphs
from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset




# -------------------------------
# GMDataset Class (as provided)
# -------------------------------
RESCALE = (320, 240)
SRC_GRAPH_CONSTRUCT = "tri"
TGT_GRAPH_CONSTRUCT = "same"
SYM_ADJACENCY = True
NORM_MEANS = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
MAX_PROB_SIZE = -1
TYPE = '2GM'
FP16 = False
RANDOM_SEED = 123
BATCH_SIZE = 8
DATALOADER_NUM = 6

train_root = 'dataset/Synthetic'
benchmark = L3SFV2AugmentedBenchmark(
        sets='test',
        obj_resize=(512, 512),
        train_root=train_root
    )



# -------------------------------
# Testing the Dataset
# -------------------------------

def main():
    # Create a dummy image file "dummy.jpg" in the current directory.
    dummy_img = np.full((240, 320, 3), 255, dtype=np.uint8)  # white image
    dummy_img_path = "dummy.jpg"
    cv2.imwrite(dummy_img_path, dummy_img)


    # Instantiate the dataset
    dataset = GMDataset(name="GMDataset", bm=benchmark, length=640, cls=None, problem="2GM")
    
    # Get one sample from the dataset
    sample = dataset[0]
    
    # Print out the keys and types in the returned sample
    print("Sample keys and types:")
    for key, value in sample.items():
        print(f"  {key}: {type(value)}")
    
    # Print shapes of the augmented images
    print("\nAugmented image shapes:")
    for i, img in enumerate(sample['images']):
        print(f"  Image {i+1}: {img.shape}")
        

    # Clean up: remove the dummy image file
    os.remove(dummy_img_path)

if __name__ == '__main__':
    main()
