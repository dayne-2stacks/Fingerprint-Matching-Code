#!/usr/bin/env python3
"""
Clean Pore Detection Script for L3-SF Dataset
Usage: python3 pore_detect_clean.py --start_index 0 --end_index 100
"""

import numpy, torch, argparse
from pathlib import Path
from util.utils import loadModel as lm
import entireImage, cv2, torchvision
from validate import readTxtList
from util.utils import plotPredImage as draw
import warnings
warnings.filterwarnings('ignore')

def detect_pores(start_idx=0, end_idx=50, device='cuda', features=40):
    """
    Detect pores in L3-SF dataset images
    
    Args:
        start_idx: Starting image index
        end_idx: Ending image index  
        device: 'cuda' or 'cpu'
        features: Number of model features (40 is default)
    """
    print(f"=== PORE DETECTION STARTING ===")
    print(f"Processing images {start_idx} to {end_idx}")
    
    # Configuration
    pathToSolution = 'out_of_the_box_detect/'
    GROUNDTRUTH = '../dataset/PolyUHRF-FP/DBII'
    imageExtension = 'jpg'
    
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
            
            # Apply NMS and save coordinates
            entireImage.apply_nms(pred, 0.65, 17, 0.2,
                                  'out_of_the_box_detect/Prediction/PolyU/Pore/',
                                  identifier,
                                  'out_of_the_box_detect/Prediction/PolyU/Coordinates/',
                                  17)
            
            # Read detections for visualization
            detections = []
            readTxtList(f'out_of_the_box_detect/Prediction/PolyU/Coordinates/{identifier}.txt',
                       detections, image.shape[0], image.shape[1], 17)
            
            # Create and save visualization
            # result_image = draw(image, detections, 5, [0, 0, 255], 1)
            cv2.imwrite(f'out_of_the_box_detect/Prediction/Fingerprint/{identifier}.png', image)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\n=== DETECTION COMPLETE ===")
    print(f"✓ Successfully processed: {processed_count} images")
    print(f"✓ Pore coordinates saved to: out_of_the_box_detect/Prediction/PolyU/Coordinates/")
    print(f"✓ Visualizations saved to: out_of_the_box_detect/Prediction/PolyU/Fingerprint/")
    print(f"✓ Binary maps saved to: out_of_the_box_detect/Prediction/PolyU/Pore/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean Pore Detection for L3-SF Dataset')
    parser.add_argument('--start_index', type=int, default=0, help='Starting image index')
    parser.add_argument('--end_index', type=int, default=50, help='Ending image index')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--features', type=int, default=40, help='Model features')
    
    args = parser.parse_args()
    detect_pores(args.start_index, args.end_index, args.device, args.features)
