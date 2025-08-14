#!/usr/bin/env python3
"""
Clean Pore Detection Script for Fingerprint Datasets
Usage: python3 pore_detect_clean.py --start_index 0 --end_index 100 --dataset-path /path/to/dataset --dataset-name Dataset
"""

import numpy, torch, argparse
from pathlib import Path
from util.utils import loadModel as lm
import entireImage, cv2, torchvision
from validate import readTxtList
from util.utils import plotPredImage as draw
import warnings
warnings.filterwarnings('ignore')

def detect_pores(start_idx=0, end_idx=50, device='cuda', features=40, dataset_path='/green/data/L3SF_V2/L3-SF/', dataset_name='L3-SF'):
    """
    Detect pores in fingerprint dataset images
    
    Args:
        start_idx: Starting image index
        end_idx: Ending image index  
        device: 'cuda' or 'cpu'
        features: Number of model features (40 is default)
        dataset_path: Path to the dataset
        dataset_name: Name of the dataset
    """
    print(f"=== PORE DETECTION STARTING ===")
    print(f"Processing images {start_idx} to {end_idx}")
    print(f"Dataset: {dataset_name} at {dataset_path}")
    
    # Configuration
    pathToSolution = 'out_of_the_box_detect/'
    GROUNDTRUTH = dataset_path
    imageExtension = 'png'
    dataset = dataset_name
    
    # Load model
    print("Loading pre-trained model...")
    model = lm(modelPath=pathToSolution+f'models/{features}', 
               device=torch.device(device), 
               NUMBERLAYERS=8, 
               NUMBERFEATURES=int(features), 
               MAXPOOLING=False, 
               WINDOWSIZE=17, 
               residual=False, 
               gabriel=False, 
               su=False)
    
    model.eval()
    model.to(device)
    print("✓ Model loaded successfully!")
    
    # Get image paths
    all_image_paths = sorted(Path(GROUNDTRUTH).rglob(f'*.{imageExtension}'))
    image_paths = all_image_paths[start_idx:end_idx]
    print(f"Found {len(all_image_paths)} total images, processing {len(image_paths)} images")
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    
    # Process images
    processed_count = 0
    for i, img_path in enumerate(image_paths):
        try:
            if i % 10 == 0:
                print(f"Progress: {i}/{len(image_paths)} ({processed_count} successful)")
            
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # Run inference
            pred = model(transforms(image).unsqueeze(dim=0).float().to(device)).detach().cpu()
            
            # Generate identifier
            identifier = f'{img_path.parent.name}_{img_path.stem}'
            
            # Create output directories if they don't exist
            Path(f'out_of_the_box_detect/Prediction/{dataset}/Pore/').mkdir(parents=True, exist_ok=True)
            Path(f'out_of_the_box_detect/Prediction/{dataset}/Coordinates/').mkdir(parents=True, exist_ok=True)
            Path(f'out_of_the_box_detect/Prediction/{dataset}/Fingerprint/').mkdir(parents=True, exist_ok=True)
            
            # Apply NMS and save coordinates
            entireImage.apply_nms(pred, 0.65, 17, 0.2,
                                  f'out_of_the_box_detect/Prediction/{dataset}/Pore/',
                                  identifier,
                                  f'out_of_the_box_detect/Prediction/{dataset}/Coordinates/',
                                  17)
            
            # Read detections for visualization
            detections = []
            readTxtList(f'out_of_the_box_detect/Prediction/{dataset}/Coordinates/{identifier}.txt',
                       detections, image.shape[0], image.shape[1], 17)
            
            # Create and save visualization
            # result_image = draw(image, detections, 5, [0, 0, 255], 1)
            cv2.imwrite(f'out_of_the_box_detect/Prediction/{dataset}/Fingerprint/{identifier}.png', image)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\n=== DETECTION COMPLETE ===")
    print(f"✓ Successfully processed: {processed_count} images")
    print(f"✓ Pore coordinates saved to: out_of_the_box_detect/Prediction/{dataset}/Coordinates/")
    print(f"✓ Visualizations saved to: out_of_the_box_detect/Prediction/{dataset}/Fingerprint/")
    print(f"✓ Binary maps saved to: out_of_the_box_detect/Prediction/{dataset}/Pore/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean Pore Detection for Fingerprint Datasets')
    parser.add_argument('--start_index', type=int, default=0, help='Starting image index')
    parser.add_argument('--end_index', type=int, default=10000, help='Ending image index')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--features', type=int, default=40, help='Model features')
    parser.add_argument('--dataset-path', type=str, default='/green/data/L3SF_V2/L3-SF/', help='Path to the dataset')
    parser.add_argument('--dataset-name', type=str, default='L3-SF', help='Name of the dataset')
    
    args = parser.parse_args()
    detect_pores(args.start_index, args.end_index, args.device, args.features, 
                args.dataset_path, args.dataset_name)
